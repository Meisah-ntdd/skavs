#!/bin/bash
#JSUB -q gpu
#JSUB -gpgpu 1
#JSUB -n 8
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -J ska
#上面的和命令行参数意义一致

#激活环境（pytorch_gpu 换成自己创建的虚拟环境，这是我创建的）
source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu

module unload cuda
module load cuda/11.6
which nvcc

#换源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple	
#pip config list

unset PYTHONPATH


#pip install -r requirements.txt

#nvidia-smi 查看显卡使用情况，结果在输出文件output，在多gpu跑代码时需要对照更改GPU序列号
#具体多GPU如何设置，见 多卡训练.txt
nvidia-smi

#下面是你自己要执行的代码

#python3 process_musan.py /home/21151213641/xdhpc 

python3 ./trainSpeakerNet.py \

        --max_frames 200 \

        --eval_frames 0 \

        --num_eval 1 \

        --num_spk 100 \

        --num_utt 2 \

        --augment Ture \

        --optimizer adamW \

        --scheduler cosine_annealing_warmup_restarts \

        --lr_t0 25 \

        --lr_tmul 1.0 \

        --lr_max 1e-3 \

        --lr_min 1e-8 \

        --lr_wstep 10 \

        --lr_gamma 0.5 \

        --margin 0.2 \

        --scale 30 \

        --num_class 87 \

        --save_path ./save/ska_tdnn \

        --train_list ./list/train_vox2.txt \

        --test_list ./list/veri_test2.txt \

        --train_path /path/to/dataset/VoxCeleb2/dev/wav \

        --test_path /path/to/dataset/VoxCeleb1/test/wav \

        --musan_path /path/to/dataset/MUSAN/musan_split \

        --rir_path /path/to/dataset/RIRS_NOISES/simulated_rirs \

        --model SKA_TDNN \

        --port 8000 \

        --distributed 

