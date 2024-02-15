# cog-aya-101
📚 Aya, an LLM by Cohere capable of understanding and generating content in 101 languages 🗣️

[![Replicate](https://replicate.com/zsxkib/aya-101/badge)](https://replicate.com/zsxkib/aya-101)


This is an implementation of [CohereForAI/aya-101](https://huggingface.co/CohereForAI/aya-101) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)


Simply run:
```sh
$ cog predict -i prompt="भारत में इतनी सारी भाषाएँ क्यों हैं?"

भारत में कई भाषाएँ हैं और विभिन्न भाषाओं के बोली जाने वाले लोग हैं। यह विभिन्नता भाषाई विविधता और सांस्कृतिक विविधता का परिणाम है।
```

This will automagically download the weights too.

⚠️ **Important Notice:**
Before attempting to run the `aya-101` model, please be aware that the model weights are extremely large. This may significantly impact download times and require substantial disk space. Ensure your system is adequately prepared to handle this load. For detailed requirements and potential impact, refer to the discussion [here](https://huggingface.co/CohereForAI/aya-101/discussions/7). This model was tested on an Nvidia A40 Large w/ 64GB RAM.