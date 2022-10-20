docker run --gpus all -it --rm -p 8891:8888 -p 6006:6006 \
-u shinhanai.kyungmin \
--name tensorflow_2_4_`date "+%H%M%s"` \
-v /home/kmkim/python_projects/LearnKit/AlgSimulation_v2:/home/shinhanai.kyungmin/AlgSimulation_v2 \
-v /home/kmkim/app_data/AlgSimulation/rawdata:/home/shinhanai.kyungmin/AlgSimulation_v2/datasets/rawdata \
-v /home/kmkim/app_data/AlgSimulation/save:/home/shinhanai.kyungmin/AlgSimulation_v2/save \
--workdir /home/shinhanai.kyungmin/AlgSimulation_v2 \
cuda_11_0/tensorflow:tensorflow-gpu_2-4-0 bash
