# nohup bash ./train_2023_10_subtaskB_bart_mnli_ep_mixed_w_partial_prompt_seed12.sh > train_2023_10_subtaskB_bart_mnli_ep_mixed_w_partial_prompt_seed12_results.log 2>&1 &

# config=../config/config-bert_mnli.txt
# config=../config/config-roberta_base_mnli_tune11.txt
# config=../config/config-xlnet_base_mnli_tune5.txt
config=../config/config-bart_large_mnli_encoder.txt

echo "start training config ${config}......"
echo "start noun phrase targets......"
# for domain in "covid19_domain" "world_event_domain" "education_and_culture_domain" "consumption_and_entertainment_domain" "sports_domain" "rights_domain" "environmental_protection_domain" "politic"
for domain in "covid19_domain" "world_event_domain" "education_and_culture_domain" "consumption_and_entertainment_domain" "sports_domain" "rights_domain" "environmental_protection_domain" "politic"
do
    
    echo "start training domain ${domain}......"
    # ###################################################################################################################
    # ###                          claim                                    
    # ###################################################################################################################
    # train_data=/data/czhao43/ez_stance/subtaskB_dataset_split/claim/${domain}/raw_train_all_onecol.csv
    # dev_data=/data/czhao43/ez_stance/subtaskB_dataset_split/claim/${domain}/raw_val_all_onecol.csv
    # test_data=/data/czhao43/ez_stance/subtaskB_dataset_split/claim/${domain}/raw_test_all_onecol.csv

    # kg_data=/data/czhao43/ez_stance/claim/raw_val_all_onecol2.csv
    # kg_data2=/data/czhao43/ez_stance/claim/raw_val_all_onecol2.csv

    # echo "start training claim targets......"
    # for seed in {0..3}
    # # for seed in {0..0}
    # do
    #     echo "start training seed ${seed}......"
    #     for epoch in {0..0}
    #     do
    #         echo "start training Gen ${epoch}......"
    #         python train_model.py -prompt_index 0 -mode train_en_test_en -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} -kg2 ${kg_data2} \
    #                               -g ${epoch} -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3  --earlystopping_step 5 -p 100
    #     done
    # done
    # ###################################################################################################################
    # ###                          mixed                               
    # ###################################################################################################################
    train_data=/data/czhao43/ez_stance/subtaskB_dataset_split/mixed/${domain}/mixed_partial_seed_12_prompt5891112/raw_train_all_onecol.csv
    dev_data=/data/czhao43/ez_stance/subtaskB_dataset_split/mixed/${domain}/mixed_partial_seed_12_prompt5891112/raw_val_all_onecol.csv
    test_data=/data/czhao43/ez_stance/subtaskB_dataset_split/mixed/${domain}/mixed_partial_seed_12_prompt5891112/raw_test_all_onecol.csv

    kg_data=/data/czhao43/ez_stance/claim/raw_val_all_onecol2.csv
    kg_data2=/data/czhao43/ez_stance/claim/raw_val_all_onecol2.csv


    echo "start training mixed targets......"
    for seed in {0..3}
    # for seed in {0..0}
    do
        echo "start training seed ${seed}......"
        for epoch in {0..0}
        do
            echo "start training Gen ${epoch}......"
            python train_model.py -prompt_index 0 -mode train_en_test_en -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} -kg2 ${kg_data2} \
                                  -g ${epoch} -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3  --earlystopping_step 5 -p 100
        done



        test_data_noun_phrase=/data/czhao43/ez_stance/subtaskB_dataset_split/noun_phrase/${domain}/noun_phrase_partial_seed_12_prompt5891112/raw_test_all_onecol.csv
        test_data_claim=/data/czhao43/ez_stance/subtaskB_dataset_split/claim/${domain}/raw_test_all_onecol.csv

        python eval_modelTest.py -prompt_index 0 -mode train_en_test_en -c ${config} -train ${train_data} -dev ${test_data_noun_phrase} -test ${test_data_claim} -kg ${kg_data}\
                              -g 0 -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3  --earlystopping_step 5 -p 100
        
    done

done

