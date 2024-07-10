data_dir="data/babel-qa"
babel_root="data/babel_v1.0_release"
amass_root="data/AMASS"
smplh_root="checkpoints/smplh"

# process data step 1
python extract_motion_concepts.py --babel_root $babel_root --data_dir $data_dir

# process data step 2
python generate_questions.py \
    --data_dir $data_dir \
    --data_split_file BABEL-QA/split_question_ids.json

# process data step 3
python BABEL-QA/process_amass_data/process_amass_data.py \
    --data_dir $data_dir \
    --babel_root $babel_root \
    --amass_root $amass_root \
    --smplh_root $smplh_root
