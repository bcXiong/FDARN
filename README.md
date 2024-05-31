# Cross-Modal Federated Human Activity Recognition via Modality-Agnostic and Modality-Specific Representation Learning


## Software requirements

- numpy, scipy, torch, Pillow, matplotlib
- To download the dependencies: **pip3 install -r requirements.txt**

- Training requires minimum 12 GB of GPU memory for batch size of 32.

## Produce experiments

- There is a main file "main.py" which allows running all experiments.
- The network is under the ```FLAlgorithms/users``` folder, marked as userfdarn.py.
- It is noted that algorithm should be run at least 5 times and then the results are averaged.
- To run training simply run
```
python main.py --dataset Epic --model dnn --batch_size 64 --learning_rate 0.001 --num_global_iters 300 --local_epochs 2 --algorithm FDARN --times 5
```
```
python main.py --dataset MM --model dnn --batch_size 32 --learning_rate 0.01 --num_global_iters 300 --local_epochs 2 --algorithm FDARN --times 5
```

```
python main.py --dataset ECM --model dnn --batch_size 32 --learning_rate 0.01 --num_global_iters 300 --local_epochs 2 --algorithm FDARN --times 5
```

```
python main.py --dataset Ego-exo --model dnn --batch_size 32 --learning_rate 0.01 --num_global_iters 300 --local_epochs 2 --algorithm FDARN --times 5
```

## Datasets

- Data should be placed under ```data/``` folder. 
- Epic-Kitchens, Multimodal-EA and Stanford-ECM are all public datasets. 
- Ego-Exo-AR dataset is available to download at: https://drive.google.com/file/d/13HvPVGQE3Lm6ovKzVCGipxAUOmDdsJU0/view?usp=sharing (To protect user privacy, we only provide image features instead of original images.)
- On the Epic-Kitchens dataset, we use 4 modalities (i.e., video, optical flow, audio and sensor) as input. 
- On the Multimodal-EA dataset, we use 2 modalities (i.e., video and sensor) as input. 
- On the Stanford-ECM dataset, we use 2 modalities (i.e., video and sensor) as input. 
- On the Ego-Exo-AR dataset, we use 2 modalities (i.e., image and sensor) as input.
- Please refer to the PDF file in Supplementary Material for details of data statistics and feature extraction.



