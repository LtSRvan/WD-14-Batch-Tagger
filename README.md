# WD-14-Batch-Tagger
This repository is a batch tagger adapted from the hugginface space that works on the swinv2 model by SmilingWolf. It will generate a txt file with the same name of the image with the prediction results inside.

It can be run on both GPU and CPU, instructions for each one below.

## Installation
To start usign clone the repository:

    git clone https://github.com/LtSRvan/WD-14-Batch-Tagger

CD into the folder:

    cd WD-14-Batch-Tagger
    
Run the **create-venv.bat** file. it will take a few minutes to finish:

    create-venv.bat

Once it's done install the **requirements.txt** in case you want to use only CPU:

    pip install -r requirements.txt

If you want to use on GPU run the **GPU-requirements.txt**:

    pip install -r GPU-requirements.txt
    
I'm still a newbie so for now you have to install one or the other, I'm working on putting it all together to avoid problems.

Dowload the [model.onnx](https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/resolve/main/model.onnx) and the [selected_tags.csv](https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/resolve/main/selected_tags.csv) and put both on the WD-14-Batch-Tagger folder

## Usage

For CPU only use:

    python wd-tagger.py --input "your/folder/path" --output "output/path"
    
For GPU use:

    python GPU-wd-tagger.py --input "your/folder/path" --output "output/path"
    
## Comand line args

### --input
Specify the directory where your images to be processed are located.

### --output 
Specify the directory where the results are going to be stored.

### --label
By default, it assumes that 'selected_tags.csv' is in the same folder as the script. If it has a different name or is on another directory use this argument to specify.

### --model
Same as '--label', by default, it assumes that 'model.onnx' is in the same folder as the script. If it has a different name or is on another directory use this argument to specify.

### --general_score
Specify the minimum 'confidence' percentage for a tag in the prediction. The lower the number, the more tags will appear, but they may be redundant or not entirely accurate to what is seen in the image.
By default it's set on 0.5
 
### --character_score
Similar to the previous one, specify the minimum 'confidence' level for a character in the prediction. The lower the number, the more likely it is that characters unrelated to the image in question may appear.
By default it's set on 0.85

## Other tweaks
If you want to use the tag format used in [MyneFactory](https://huggingface.co/MyneFactory), you can modify line 82 (87 on the GPU one):

    return f"{a}\n"
    
To:

    return f"Inster your text here, {a}\n"
    
For example (using the [MF-Bofuri](https://huggingface.co/MyneFactory/MF-Bofuri) for reference):

    return f"BoMaple, {a}\n"
    
Keep in mind that to use it this way, you must have your dataset well organized by characters/artists/style so that it does not 'contaminate' other images that do not require that specific addition.
    



