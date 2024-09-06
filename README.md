# EZ-STANCE
This repository includes data and code for the EZ-STANCE dataset (recently accepted as an ACL 2024 main conference paper).

Please download our dataset using this link: [data](https://drive.google.com/file/d/1WxysM2VwjcZkHnCDMe7P9bwA8VW7IU01/view?usp=drive_link)


**Here are some detailed instructions on the data:**

1.  In our experiment, we used the 'Text' column as the text, which I preprocessed based on the original raw tweets. This preprocessing included removing spaces, emojis, @, #, and other special characters. The unprocessed text was placed in the 'Ori Text' column.

2. The 'seen' column was retained according to vast, but it is not very useful because 'train' is always 1, while 'val' and 'test' are always 0.

3. For the 'subtaskB' train and val sets, we need to filter the 'In Use' column during actual training, using only the entries where 'In Use' == 1. This is because, during data splitting for subtaskB, one domain is chosen as the test set, and the remaining seven domains may have targets that overlap with the test domain. These overlapping targets need to be masked. The 'in use' column is used for this masking.

4. In our paper, we proposed a method that uses MNLI pre-trained models for training. In this case, the noun phrases are prompted (which also affects 'mixed' because it also contains noun-phrase targets). Therefore, in the paths corresponding to 'noun phrase' and 'mixed', there is a 'prompt' folder that contains the data with the prompted noun phrase targets. Results based on NLI prompting (e.g., **BART-MNLI-ep**) were obtained using these data.

We will upload our code soon.

Please cite us at:
```
@inproceedings{zhao-caragea-2024-ez,
    title = "{EZ}-{STANCE}: A Large Dataset for {E}nglish Zero-Shot Stance Detection",
    author = "Zhao, Chenye  and
      Caragea, Cornelia",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.838",
    pages = "15697--15714",
    abstract = "Zero-shot stance detection (ZSSD) aims to determine whether the author of a text is in favor, against, or neutral toward a target that is unseen during training. In this paper, we present EZ-STANCE, a large English ZSSD dataset with 47,316 annotated text-target pairs. In contrast to VAST, which is the only other large existing ZSSD dataset for English, EZ-STANCE is 2.5 times larger, includes both noun-phrase targets and claim targets that cover a wide range of domains, provides two challenging subtasks for ZSSD: target-based ZSSD and domain-based ZSSD, and contains much harder examples for the neutral class. We evaluate EZ-STANCE using state-of-the-art deep learning models. Furthermore, we propose to transform ZSSD into the NLI task by applying simple yet effective prompts to noun-phrase targets. Our experimental results show that EZ-STANCE is a challenging new benchmark, which provides significant research opportunities on English ZSSD. We publicly release our dataset and code at https://github.com/chenyez/EZ-STANCE.",
}
```
