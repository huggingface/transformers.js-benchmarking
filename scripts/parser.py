import io
import logging
import requests
from requests.adapters import HTTPAdapter
import onnx
from google.protobuf.internal.encoder import _VarintBytes
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(message)s")

MIN_CHUNK_SIZE = 8 * 1024           # 8 KB minimum per Range request to batch headers
WANTED_TAGS    = {1, 2, 11, 12, 13} # GraphProto: node(1), name(2), input(11), output(12), value metadata(13)

class LoggingRetry(Retry):
    def increment(self, method, url, response=None, error=None, _pool=None, _stacktrace=None):
        new_retry = super().increment(method, url, response, error, _pool, _stacktrace)
        attempt = (self.total - new_retry.total + 1) if self.total is not None else 'unknown'
        delay = new_retry.get_backoff_time()
        err_str = f" Error: {error}" if error else ""
        sleep_str = f" Sleeping for {delay:.2f} seconds." if delay > 0 else ""
        logging.info(f"Retrying {method} request to {url}. Attempt {attempt}.{err_str}{sleep_str}")
        return new_retry

class RangeFetcher:
    def __init__(self, url, retries=10, backoff_factor=2, timeout=10):
        self.url = url
        self.session = requests.Session()
        self.buffer = bytearray()
        self.loaded_ranges = []   # list of (start, end), inclusive
        self.total_downloaded = 0
        self.timeout = timeout

        retry_strategy = LoggingRetry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        retry_strategy.logger = logging.getLogger("urllib3.retry")
        retry_strategy.logger.setLevel(logging.INFO)
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _add_range(self, start, data):
        """Insert `data` at offset `start` into self.buffer, merge loaded_ranges."""
        end = start + len(data) - 1
        if end >= len(self.buffer):
            self.buffer.extend(b'\x00' * (end + 1 - len(self.buffer)))
        self.buffer[start:end+1] = data

        # merge intervals
        new = (start, end)
        merged = []
        i = 0
        while i < len(self.loaded_ranges) and self.loaded_ranges[i][1] < new[0] - 1:
            merged.append(self.loaded_ranges[i]); i += 1
        while i < len(self.loaded_ranges) and self.loaded_ranges[i][0] <= new[1] + 1:
            new = (min(new[0], self.loaded_ranges[i][0]),
                   max(new[1], self.loaded_ranges[i][1]))
            i += 1
        merged.append(new)
        merged.extend(self.loaded_ranges[i:])
        self.loaded_ranges = merged

    def fetch(self, start, end=None):
        """
        Ensure buffer[start..end] is loaded (inclusive).
        If end=None, do one fetch from `start` to EOF (no min-size).
        Otherwise, fetch only the missing sub-ranges, each >= MIN_CHUNK_SIZE.
        """
        if end is None:
            # open‐ended final fetch
            headers = {"Range": f"bytes={start}-"}
            logging.info(f"Requesting bytes {start}-<EOF>")
            resp = self.session.get(self.url, headers=headers); resp.raise_for_status()
            chunk = resp.content
            self.total_downloaded += len(chunk)
            logging.info(f" → {len(chunk)} bytes  (total {self.total_downloaded})")
            self._add_range(start, chunk)
            return self.buffer[start:start+len(chunk)]

        # find holes in [start..end]
        to_fetch = []
        cursor = start
        for (a, b) in self.loaded_ranges:
            if b < cursor: continue
            if a > end:   break
            if a > cursor:
                to_fetch.append((cursor, a-1))
            cursor = max(cursor, b+1)
        if cursor <= end:
            to_fetch.append((cursor, end))

        # fetch each hole (enforcing MIN_CHUNK_SIZE)
        for (s, e) in to_fetch:
            length = e - s + 1
            if length < MIN_CHUNK_SIZE:
                e = s + MIN_CHUNK_SIZE - 1
            headers = {"Range": f"bytes={s}-{e}"}
            logging.info(f"Requesting bytes {s}-{e}")
            resp = self.session.get(self.url, headers=headers); resp.raise_for_status()
            chunk = resp.content
            self.total_downloaded += len(chunk)
            logging.info(f" → {len(chunk)} bytes  (total {self.total_downloaded})")
            self._add_range(s, chunk)

        return self.buffer[start:end+1]

# Based on https://raw.githubusercontent.com/onnx/onnx/refs/heads/main/onnx/onnx.proto

def read_varint(stream):
    """Read a Base-128 varint from `stream`. Returns (value, bytes_read)."""
    result = 0; shift = 0; count = 0
    while True:
        b = stream.read(1)
        if not b: raise EOFError
        b = b[0]
        result |= (b & 0x7F) << shift
        count += 1
        if not (b & 0x80): break
        shift += 7
    return result, count

def skip_field(stream, wire_type):
    """Advance `stream` past the next field of the given `wire_type`. Returns skipped bytes."""
    if wire_type == 0:       # varint
        _, n = read_varint(stream); return n
    elif wire_type == 1:     # fixed64
        stream.seek(8, io.SEEK_CUR); return 8
    elif wire_type == 2:     # length-delimited
        length, n = read_varint(stream)
        stream.seek(length, io.SEEK_CUR)
        return n + length
    elif wire_type == 5:     # fixed32
        stream.seek(4, io.SEEK_CUR); return 4
    else:
        raise ValueError(f"Unsupported wire type {wire_type}")

def locate_graph_header(fetcher, probe_size=1*1024*1024):
    """
    Download the first `probe_size` bytes, scan for ModelProto.graph (field 7, wire 2).
    Returns (before_bytes, graph_payload_offset, graph_length).
    """
    data = fetcher.fetch(0, probe_size-1)
    stream = io.BytesIO(data)
    before = bytearray()

    while True:
        key_pos = stream.tell()
        try:
            key, key_len = read_varint(stream)
        except EOFError:
            raise RuntimeError("Couldn't find graph header in initial bytes")
        field_num, wire_type = key >> 3, key & 0x7

        if field_num == 7 and wire_type == 2:
            length, length_len = read_varint(stream)
            header_len = key_len + length_len
            payload_off = key_pos + header_len
            return data[:key_pos], payload_off, length

        # skip and accumulate everything before
        skipped = skip_field(stream, wire_type)
        stream.seek(key_pos + key_len + skipped)
        before.extend(data[key_pos:key_pos + key_len + skipped])

def extract_graph_structure(fetcher, graph_off, graph_len):
    """
    Walk the GraphProto payload [graph_off..graph_off+graph_len) and
    collect only fields whose tag in WANTED_TAGS, skipping all others.
    Returns the concatenated bytes of just those wanted fields.
    """
    out = bytearray()
    pos = graph_off
    end = graph_off + graph_len

    while pos < end:
        # fetch minimal header (varint key + possible length prefix)
        hdr = fetcher.fetch(pos, pos + 19)
        hdr_stream = io.BytesIO(hdr)
        key, key_len = read_varint(hdr_stream)
        field_num, wire_type = key >> 3, key & 0x7

        length = 0; length_len = 0
        if wire_type == 2:
            length, length_len = read_varint(hdr_stream)

        # compute total bytes for this field
        if wire_type == 2:
            total = key_len + length_len + length
        elif wire_type == 0:
            total = key_len + skip_field(io.BytesIO(hdr[key_len:]), wire_type)
        elif wire_type == 1:
            total = key_len + 8
        elif wire_type == 5:
            total = key_len + 4
        else:
            raise ValueError(f"Bad wire type {wire_type}")

        # if it’s not one of the wanted tags, skip it entirely
        if field_num not in WANTED_TAGS:
            pos += total
            continue

        # otherwise fetch & append it in one go
        chunk = fetcher.fetch(pos, pos + total - 1)
        out.extend(chunk)
        pos += total

    return bytes(out)

def stream_parse_model_header(url):
    fetcher = RangeFetcher(url)

    # 1) find graph payload
    before, graph_off, graph_len = locate_graph_header(fetcher)

    # 2) stream‐parse only node/name/input/output fields
    graph_struct = extract_graph_structure(fetcher, graph_off, graph_len)

    # 3) fetch any ModelProto fields after graph (small headers, metadata)
    after = fetcher.fetch(graph_off + graph_len, None)

    # 4) rebuild minimal ModelProto
    tag = _VarintBytes((7 << 3) | 2)
    length = _VarintBytes(len(graph_struct))
    model_bytes = before + tag + length + graph_struct + after

    # 5) parse with ONNX
    model = onnx.ModelProto()
    model.ParseFromString(model_bytes)

    logging.info(f"Total bytes downloaded: {fetcher.total_downloaded}")
    return model
