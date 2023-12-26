# deepnwp

$ docker run --gpus all -it -v /home/cccr_rnd/manmeet/deepnwp:/apollo  -p 8891:8891  manmeet3591/deepnwp:v6

$ jupyter-notebook --ip 0.0.0.0 --port=8891 --no-browser --allow-root &

$ ssh -N -f -L localhost:8891:localhost:8891 cccr_rnd@10.12.1.28
