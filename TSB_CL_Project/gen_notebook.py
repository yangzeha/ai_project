import json
import os

def generate_notebook():
    # 1. Define Cell Content
    
    # Cell 1: Setup
    source_1 = [
        "# [1] Setup Environment & Clone Repo\n",
        "import os\n",
        "if os.path.exists('/kaggle/working'):\n",
        "    os.chdir('/kaggle/working')\n",
        "    !rm -rf ai_project\n",
        "\n",
        "!git clone https://github.com/yangzeha/ai_project.git\n",
        "%cd ai_project\n",
        "!pip install -r TSB_CL_Project/requirements.txt\n",
        "!apt-get update && apt-get install -y libsparsehash-dev\n"
    ]
    
    # Cell 2: Compile
    source_2 = [
        "# [2] Compile MSBE (C++)\n",
        "!g++ -O3 Similar-Biclique-Idx-main/main.cpp -o Similar-Biclique-Idx-main/msbe\n",
        "!chmod +x Similar-Biclique-Idx-main/msbe\n"
    ]
    
    # Cell 3: Prepare Data Script (Using Python write instead of magic)
    # Read local file content
    with open(r"c:\Users\LENOVO\Desktop\论文代码\ai_project\TSB_CL_Project\prepare_yelp2018.py", "r", encoding="utf-8") as f:
        prepare_content = f.read()
    
    # Escape backslashes to prevent them from being interpreted as escape sequences in the python string
    prepare_content = prepare_content.replace('\\', '\\\\')
    # Escape triple quotes
    prepare_content = prepare_content.replace('"""', '\\"\\"\\"')
    
    source_3 = [
        "# [3] Create Data Preparation Script\n",
        "import os\n",
        "code = \"\"\"\n"
    ] + [line + "\n" for line in prepare_content.splitlines()] + [
        "\"\"\"\n",
        "os.makedirs('TSB_CL_Project', exist_ok=True)\n",
        "with open('TSB_CL_Project/prepare_yelp2018.py', 'w', encoding='utf-8') as f:\n",
        "    f.write(code)\n"
    ]
    
    # Cell 4: Run Prepare
    source_4 = [
        "# [4] Download & Process Yelp2018\n",
        "!python TSB_CL_Project/prepare_yelp2018.py\n"
    ]
    
    # Cell 5: Proof Script (Using Python write instead of magic)
    with open(r"c:\Users\LENOVO\Desktop\论文代码\ai_project\TSB_CL_Project\quick_proof_yelp.py", "r", encoding="utf-8") as f:
        proof_content = f.read()
        
    # Escape backslashes
    proof_content = proof_content.replace('\\', '\\\\')
    # Escape triple quotes
    proof_content = proof_content.replace('"""', '\\"\\"\\"')
        
    source_5 = [
        "# [5] Create Quick Proof Script\n",
        "import os\n",
        "code = \"\"\"\n"
    ] + [line + "\n" for line in proof_content.splitlines()] + [
        "\"\"\"\n",
        "os.makedirs('TSB_CL_Project', exist_ok=True)\n",
        "with open('TSB_CL_Project/quick_proof_yelp.py', 'w', encoding='utf-8') as f:\n",
        "    f.write(code)\n"
    ]
    
    # Cell 6: Run Proof
    source_6 = [
        "# [6] Run Quick Proof (Static TSB-CL vs LightGCN)\n",
        "!python TSB_CL_Project/quick_proof_yelp.py\n"
    ]
    
    # 2. Construct Notebook JSON
    notebook = {
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_1},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_2},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_3},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_4},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_5},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_6}
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # 3. Write to file
    output_path = r"c:\Users\LENOVO\Desktop\论文代码\ai_project\Run_on_Kaggle.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Notebook generated at {output_path}")

if __name__ == "__main__":
    generate_notebook()
