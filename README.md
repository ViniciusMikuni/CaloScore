# Caloscore official repository


# Run the training scripts with

```bash
cd scripts
python train.py   --config CONFIG --model MODEL
```
MODEL options are: subVPSDE/VESDE/VPSDE
CONFIG options are ```config_dataset1.json/config_dataset2.json/config_dataset3.json```

#Producing new samples

```bash
python plot_caloscore.py  --nevts N  --sample  --config CONFIG --model MODEL
```
to plot  the high level distributions used in the paper call the same script as

```bash
python plot_caloscore.py  --config CONFIG --model MODEL --nslices 1
```


