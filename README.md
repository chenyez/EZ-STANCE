# EZ-STANCE
This repository includes data and code for the EZ-STANCE dataset (recently accepted as an ACL 2024 main conference paper).

Please download our dataset using this link: [data](https://drive.google.com/file/d/1WxysM2VwjcZkHnCDMe7P9bwA8VW7IU01/view?usp=drive_link)


**Here are some detailed instructions on the data:**

1.  In our experiment, we used the 'Text' column as the text, which I preprocessed based on the original raw tweets. This preprocessing included removing spaces, emojis, @, #, and other special characters. The unprocessed text was placed in the 'Ori Text' column.

2. The 'seen' column was retained according to vast, but it is not very useful because 'train' is always 1, while 'val' and 'test' are always 0.

3. For the 'subtaskB' train and val sets, we need to filter the 'In Use' column during actual training, using only the entries where 'In Use' == 1. This is because, during data splitting for subtaskB, one domain is chosen as the test set, and the remaining seven domains may have targets that overlap with the test domain. These overlapping targets need to be masked. The 'in use' column is used for this masking.

4. In our paper, we proposed a method that uses MNLI pre-trained models for training. In this case, the noun phrases are prompted (which also affects 'mixed' because it also contains noun-phrase targets). Therefore, in the paths corresponding to 'noun phrase' and 'mixed', there is a 'prompt' folder that contains the data with the prompted noun phrase targets. Results based on NLI prompting (e.g., **BART-MNLI-ep**) were obtained using these data.

We will upload our code soon.
