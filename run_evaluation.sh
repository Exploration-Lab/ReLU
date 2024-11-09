# tofu_base_path=<tofu base path goes here> (removed for privacy)

model_paths=(
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625/grad_ascent_1e-05_forget05_5/checkpoint-6
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625/grad_ascent_1e-05_forget05_5/checkpoint-12
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625/grad_ascent_1e-05_forget05_5/checkpoint-18
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625/grad_ascent_1e-05_forget05_5/checkpoint-24
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625/grad_ascent_1e-05_forget05_5/checkpoint-30
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625/grad_diff_1e-05_forget05_5/checkpoint-6
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625/grad_diff_1e-05_forget05_5/checkpoint-12
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625/grad_diff_1e-05_forget05_5/checkpoint-18
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625/grad_diff_1e-05_forget05_5/checkpoint-24
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625/grad_diff_1e-05_forget05_5/checkpoint-30
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625/KL_1e-05_forget05_5/checkpoint-6
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625/KL_1e-05_forget05_5/checkpoint-12
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625/KL_1e-05_forget05_5/checkpoint-18
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625/KL_1e-05_forget05_5/checkpoint-24
    ${tofu_base_path}/tofu/paper_models/final_ft_noLORA_5_epochs_inst_lr2e-05_phi_full/checkpoint-625/KL_1e-05_forget05_5/checkpoint-30
    )


dataset_paths_mcqa=(
                # "./data_reformat/manual/forget01.json" 
                # "./data_reformat/manual/retain99.json"
                "./data_reformat/manual/forget05.json"
                "./data_reformat/manual/retain95.json" 
                # "./data_reformat/manual/forget10.json"
                # "./data_reformat/manual/retain90.json"
                )

dataset_path_cloze=(
                # "./data_reformat/cloze/forget01_cloze.json" 
                # "./data_reformat/cloze/retain99_cloze.json"
                "./data_reformat/cloze/forget05_cloze.json"
                "./data_reformat/cloze/retain95_cloze.json" 
                # "./data_reformat/cloze/forget10_cloze.json"
                # "./data_reformat/cloze/retain90_cloze.json"
                )

dataset_path_comprehension=(
                # "./data_reformat/comprehension/forget01.json" 
                # "./data_reformat/comprehension/retain99.json"
                "./data_reformat/comprehension/forget05.json"
                "./data_reformat/comprehension/retain95.json" 
                # "./data_reformat/comprehension/forget10.json"
                # "./data_reformat/comprehension/retain90.json"
                )

dataset_path_analogy_mcqa=(
                # "./data_reformat/analogy/mcqa/forget01.json" 
                # "./data_reformat/analogy/mcqa/retain99.json"
                "./data_reformat/analogy/mcqa/forget05.json"
                "./data_reformat/analogy/mcqa/retain95.json" 
                # "./data_reformat/analogy/mcqa/forget10.json"
                # "./data_reformat/analogy/mcqa/retain90.json"
                )

dataset_path_odd_one_out=(
                # "./data_reformat/OddOneOut/odd_one_forget01.json" 
                # "./data_reformat/OddOneOut/odd_one_retain99.json"
                "./data_reformat/OddOneOut/odd_one_forget05.json"
                # "./data_reformat/OddOneOut/odd_one_retain95.json" 
                # "./data_reformat/OddOneOut/odd_one_forget10.json"
                # "./data_reformat/OddOneOut/odd_one_retain90.json"
                )

dataset_path_qualitative=(
                "./data_reformat/qualitative/match05.json"
                "./data_reformat/qualitative/match95.json"
                )

# evaluation over the mcqa format
python evaluate.py --dataset_path="./data_reformat/manual/forget05.json" --reformat_type="mcqa" --use_pretrained
python evaluate.py --dataset_path="./data_reformat/manual/retain95.json" --reformat_type="mcqa" --use_pretrained
list_num_options=(2 3 4)
for num_options in "${list_num_options[@]}"; do
    for model_path in  "${model_paths[@]}"; do
        for dataset_path in "${dataset_paths_mcqa[@]}"; do
            python evaluate.py --dataset_path=$dataset_path --num_options=$num_options --model_path=$model_path --reformat_type="cloze"
        done
    done
done

# evluation over the cloze format
python evaluate.py --dataset_path="./data_reformat/cloze/forget05_cloze.json" --reformat_type="cloze" --use_pretrained
python evaluate.py --dataset_path="./data_reformat/cloze/retain95_cloze.json" --reformat_type="cloze" --use_pretrained
for dataset_path in "${dataset_path_cloze[@]}"; do
    for model_path in  "${model_paths[@]}"; do
        python evaluate.py --dataset_path=$dataset_path --model_path=$model_path --reformat_type="cloze"
    done
done

# evaluation over comprehension-mcqa format
CUDA_VISIBLE_DEVICES=3 python evaluate.py --dataset_path="./data_reformat/comprehension/forget05.json" --reformat_type="comprehension-mcqa" --use_pretrained

list_num_options=(4 3 2)
for num_options in "${list_num_options[@]}"; do
    for dataset_path in "${dataset_path_comprehension[@]}"; do
        for model_path in  "${model_paths[@]}"; do
            python evaluate.py --dataset_path=$dataset_path --num_options=$num_options --model_path=$model_path --reformat_type="comprehension-mcqa"
        done
    done
done

# evaluation over comprehension-qa format
python evaluate.py --dataset_path="./data_reformat/comprehension/forget05.json" --reformat_type="comprehension-qa" --use_pretrained
python evaluate.py --dataset_path="./data_reformat/comprehension/retain95.json" --reformat_type="comprehension-qa" --use_pretrained


for dataset_path in "${dataset_path_comprehension[@]}"; do
    for model_path in  "${model_paths[@]}"; do
        python evaluate.py --dataset_path=$dataset_path --model_path=$model_path --reformat_type="comprehension-qa"
    done
done

# evaluation over analogy format
python evaluate.py --dataset_path="./data_reformat/analogy/mcqa/forget05.json" --reformat_type="analogy" --use_pretrained

list_num_options=(4 3 2)
for num_options in "${list_num_options[@]}"; do
    for dataset_path in "${dataset_path_analogy_mcqa[@]}"; do
        for model_path in  "${model_paths[@]}"; do
            python evaluate.py --dataset_path=$dataset_path --num_options=$num_options --model_path=$model_path --reformat_type="analogy-mcqa"
        done
    done
done


# evaluation over odd-one-out format
python evaluate.py --dataset_path="./data_reformat/OddOneOut/odd_one_forget05.json" --reformat_type="odd-one-out" --use_pretrained

list_num_options=(4 3 2)
for num_options in "${list_num_options[@]}"; do
    for dataset_path in "${dataset_path_odd_one_out[@]}"; do
        for model_path in  "${model_paths[@]}"; do
            python evaluate.py --dataset_path=$dataset_path --num_options=$num_options --model_path=$model_path --reformat_type="odd-one-out"
        done
    done
done


# qualitative analysis

python evaluate.py --dataset_path="./data_reformat/qualitative/match95.json" --use_pretrained --qualitative_analysis


for dataset_path in "${dataset_path_qualitative[@]}"; do
    for model_path in  "${model_paths[@]}"; do
        python evaluate.py --dataset_path=$dataset_path --model_path=$model_path --qualitative_analysis
    done
done
