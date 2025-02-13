export default {
  name: "Translation",
  config: {
    task: "translation",
    model_id: "Xenova/tiny-random-M2M100ForConditionalGeneration",
  },
  tests: [
    {
      name: "Default",
      inputs: [
        "जीवन एक चॉकलेट बॉक्स की तरह है।",
        {
          src_lang: "hi",
          tgt_lang: "fr",
          max_new_tokens: 5,
        },
      ],
      expected: [{ translation_text: "Slovenska төсли төсли төсли" }],
    },
  ],
};
