# nohup bash ./run_bart_mnli_subtaskA.sh > run_bart_mnli_subtaskA_results.log 2>&1 &

# config=../config/config-xlnet_base_tune5.txt
# config=../config/config-bert_mnli.txt
# config=../config/config-roberta_base_mnli.txt
# config=../config/config-xlnet_base_mnli.txt
for config in "../config/config-bart_large_mnli_encoder.txt"
do
    echo "start training config ${config}......"
    ###################################################################################################################
    ###                          claim                                    
    ###################################################################################################################
    train_data=/data/czhao43/ez_stance/subtaskA_dataset_split/claim/combined_easy/raw_train_all_onecol.csv
    dev_data=/data/czhao43/ez_stance/subtaskA_dataset_split/claim/combined_easy/raw_val_all_onecol.csv
    test_data=/data/czhao43/ez_stance/subtaskA_dataset_split/claim/combined_easy/raw_test_all_onecol.csv

    kg_data=/data/czhao43/ez_stance/claim/raw_val_all_onecol2.csv
    kg_data2=/data/czhao43/ez_stance/claim/raw_val_all_onecol2.csv


    for seed in {0..3}
    do
        echo "start training seed ${seed}......"
        for epoch in {0..0}
        do
            echo "start training Gen ${epoch}......"
            python train_model.py -prompt_index 0 -mode train_en_test_en -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} -kg2 ${kg_data2} \
                                  -g ${epoch} -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3  --earlystopping_step 5 -p 100
        done


        test_data_noun_phrase=/data/czhao43/ez_stance/subtaskA_dataset_split/noun_phrase/raw_test_all_onecol.csv
        test_data_claim=/data/czhao43/ez_stance/subtaskA_dataset_split/mixed/combined_easy/raw_test_all_onecol.csv

        python eval_modelTest.py -prompt_index 0 -mode train_en_test_en -c ${config} -train ${train_data} -dev ${test_data_noun_phrase} -test ${test_data_claim} -kg ${kg_data}\
                              -g 0 -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3  --earlystopping_step 5 -p 100
    done



    # ###################################################################################################################
    # ###                          noun phrase                                    
    # ###################################################################################################################
    train_data=/data/czhao43/ez_stance/subtaskA_dataset_split/noun_phrase/raw_train_all_onecol.csv
    dev_data=/data/czhao43/ez_stance/subtaskA_dataset_split/noun_phrase/raw_val_all_onecol.csv
    test_data=/data/czhao43/ez_stance/subtaskA_dataset_split/noun_phrase/raw_test_all_onecol.csv

    kg_data=/data/czhao43/ez_stance/claim/raw_val_all_onecol2.csv
    kg_data2=/data/czhao43/ez_stance/claim/raw_val_all_onecol2.csv


    for seed in {0..3}
    do
        echo "start training seed ${seed}......"
        for epoch in {0..0}
        do
            echo "start training Gen ${epoch}......"
            python train_model.py -prompt_index 0 -mode train_en_test_en -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} -kg2 ${kg_data2} \
                                  -g ${epoch} -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3  --earlystopping_step 5 -p 100
        done


        test_data_noun_phrase=/data/czhao43/ez_stance/subtaskA_dataset_split/claim/combined_easy/raw_test_all_onecol.csv
        test_data_claim=/data/czhao43/ez_stance/subtaskA_dataset_split/mixed/combined_easy/raw_test_all_onecol.csv

        python eval_modelTest.py -prompt_index 0 -mode train_en_test_en -c ${config} -train ${train_data} -dev ${test_data_noun_phrase} -test ${test_data_claim} -kg ${kg_data}\
                              -g 0 -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3  --earlystopping_step 5 -p 100
        
    done
done

for config in "../config/config-bart_large_mnli_encoder.txt"
do
    echo "start training config ${config}......"
    ###################################################################################################################
    ###                          noun phrase                                    
    ###################################################################################################################
    train_data=/data/czhao43/ez_stance/subtaskA_dataset_split/noun_phrase/noun_phrase_partial_seed_2_prompt5891112/raw_train_all_onecol.csv
    dev_data=/data/czhao43/ez_stance/subtaskA_dataset_split/noun_phrase/noun_phrase_partial_seed_2_prompt5891112/raw_val_all_onecol.csv
    test_data=/data/czhao43/ez_stance/subtaskA_dataset_split/noun_phrase/noun_phrase_partial_seed_2_prompt5891112/raw_test_all_onecol.csv

    kg_data=/data/czhao43/ez_stance/claim/raw_val_all_onecol2.csv
    kg_data2=/data/czhao43/ez_stance/claim/raw_val_all_onecol2.csv


    for seed in {0..3}
    do
        echo "start training seed ${seed}......"
        for epoch in {0..0}
        do
            echo "start training Gen ${epoch}......"
            python train_model.py -prompt_index 0 -mode train_en_test_en -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} -kg2 ${kg_data2} \
                                  -g ${epoch} -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3  --earlystopping_step 5 -p 100
        done


        test_data_noun_phrase=/data/czhao43/ez_stance/subtaskA_dataset_split/claim/combined_easy/raw_test_all_onecol.csv
        test_data_claim=/data/czhao43/ez_stance/subtaskA_dataset_split/mixed/combined_easy/raw_test_all_onecol.csv

        python eval_modelTest.py -prompt_index 0 -mode train_en_test_en -c ${config} -train ${train_data} -dev ${test_data_noun_phrase} -test ${test_data_claim} -kg ${kg_data}\
                              -g 0 -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3  --earlystopping_step 5 -p 100
    done
done



