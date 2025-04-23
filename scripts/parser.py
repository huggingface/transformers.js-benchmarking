import io
import requests
import onnx
from google.protobuf.internal.encoder import _VarintBytes
from google.protobuf.message import DecodeError

MB = 1024 * 1024

# --- Helpers for reading varints and fields from a binary stream ---
# Based on https://raw.githubusercontent.com/onnx/onnx/refs/heads/main/onnx/onnx.proto

def read_varint(stream):
    """Reads a varint from stream and returns (value, bytes_read)."""
    result = 0
    shift = 0
    bytes_read = 0
    while True:
        b = stream.read(1)
        if not b:
            raise EOFError("Unexpected EOF while reading varint")
        b = b[0]
        result |= (b & 0x7F) << shift
        bytes_read += 1
        if not (b & 0x80):
            break
        shift += 7
    return result, bytes_read

def parse_field(stream):
    """
    Reads one key/value field from stream and returns:
      (field_number, wire_type, raw_bytes, total_bytes)
    """
    start = stream.tell()
    try:
        key, key_bytes = read_varint(stream)
    except EOFError:
        return None, None, None, 0
    field_number = key >> 3
    wire_type = key & 0x07

    if wire_type == 0:  # varint
        _, value_bytes = read_varint(stream)
        total = key_bytes + value_bytes
    elif wire_type == 1:  # 64-bit
        stream.read(8)
        total = key_bytes + 8
    elif wire_type == 2:  # length-delimited
        length, length_bytes = read_varint(stream)
        stream.read(length)
        total = key_bytes + length_bytes + length
    elif wire_type == 5:  # 32-bit
        stream.read(4)
        total = key_bytes + 4
    else:
        raise ValueError(f"Unsupported wire type: {wire_type}")
    stream.seek(start)
    raw = stream.read(total)
    return field_number, wire_type, raw, total

# --- Functions for extracting and filtering the graph field ---

def extract_graph_field(model_bytes):
    """
    Given a ModelProto binary (from the beginning of the file), scan field by field
    until the graph field (number 7) is found. Return a tuple:
       (bytes_before_graph, graph_field_raw, bytes_after_graph)
    """
    stream = io.BytesIO(model_bytes)
    before = bytearray()
    found_graph = False
    while stream.tell() < len(model_bytes):
        pos = stream.tell()
        field_number, wire_type, raw, consumed = parse_field(stream)

        if field_number is None:
            break
        stream.seek(pos + consumed)
        if field_number == 7:
            # This is the graph field.
            graph_field = raw
            found_graph = True
            break
        else:
            before.extend(raw)
    if not found_graph:
        raise ValueError("Graph field (field number 7) not found in ModelProto")
    after = model_bytes[stream.tell():]
    return bytes(before), graph_field, bytes(after)

def filter_graph_bytes(graph_field_raw):
    """
    The graph field raw bytes have the format:
      tag (varint) + length (varint) + graph_bytes
    This function extracts the inner graph_bytes and then scans through its fields.
    It omits fields with number 5 (initializer) and 15 (sparse_initializer).
    """
    stream = io.BytesIO(graph_field_raw)
    # Read the tag and length for the graph field.
    tag, tag_bytes = read_varint(stream)
    length, length_bytes = read_varint(stream)
    graph_bytes = stream.read(length)

    # Now filter the GraphProto: keep all fields except initializer (5) and sparse_initializer (15).
    graph_stream = io.BytesIO(graph_bytes)
    output = bytearray()
    while graph_stream.tell() < len(graph_bytes):
        pos = graph_stream.tell()
        field_number, wire_type, raw, consumed = parse_field(graph_stream)
        graph_stream.seek(pos + consumed)
        if field_number in (5, 15):
            # Skip initializer fields (which contain heavy tensor data).
            continue
        output.extend(raw)
    return bytes(output)

def reconstruct_model_bytes(before, new_graph_bytes, after):
    """
    Reconstruct a new ModelProto binary:
      <before fields> + <graph field tag & new length & new_graph_bytes> + <after fields>
    The graph field tag is constructed for field number 7 and wire type 2.
    """
    tag = _VarintBytes((7 << 3) | 2)
    length = _VarintBytes(len(new_graph_bytes))
    graph_field = tag + length + new_graph_bytes
    return before + graph_field + after

def try_parse_buffer(buffer):
    # Try extracting the required bytes (without initializers)
    before, graph_field_raw, after = extract_graph_field(buffer)
    filtered_graph = filter_graph_bytes(graph_field_raw)
    new_model_bytes = reconstruct_model_bytes(before, filtered_graph, after)

    # Successfully extracted everything we need!
    model = onnx.ModelProto()
    model.ParseFromString(new_model_bytes)
    return model  # Return the graph-only model

def stream_parse_model_header(url, initial_chunk_size=1 * MB):
    """
    Stream in the ONNX model, extracting the header and graph structure.
    Downloads the minimum necessary bytes dynamically.
    """
    total_loaded = 0
    buffer = bytearray()
    session = requests.Session()
    
    def fetch_range(start, end):
        """Fetch a specific byte range from the ONNX file."""
        headers = {"Range": f"bytes={start}-{end}"}
        response = session.get(url, headers=headers)
        response.raise_for_status()
        return response.content

    # Start with an initial chunk
    chunk = fetch_range(0, initial_chunk_size - 1)
    buffer.extend(chunk)
    total_loaded += len(chunk)

    while True:
        try:
            return try_parse_buffer(buffer)

        except (ValueError, DecodeError, EOFError) as e:
            # If the graph field wasn't fully found yet, request more data (4 MB).
            next_chunk_size = 4 * MB
            end_range = total_loaded + next_chunk_size - 1
            if end_range > 40 * MB:
                raise RuntimeError("Model file is too large to parse.")
            next_chunk = fetch_range(total_loaded, end_range)
            buffer.extend(next_chunk)
            total_loaded += len(next_chunk)

            if not next_chunk:  # Stop if no more data is available
                raise RuntimeError("Unexpected end of file while parsing ONNX model.")

