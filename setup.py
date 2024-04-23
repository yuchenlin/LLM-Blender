from setuptools import setup, find_packages

setup(
    name='llm_blender',
    version='0.0.2',
    description='LLM-Blender, an innovative ensembling framework to attain consistently superior performance by leveraging the diverse strengths and weaknesses of multiple open-source large language models (LLMs). LLM-Blender cut the weaknesses through ranking and integrate the strengths through fusing generation to enhance the capability of LLMs.',
    author='Dongfu Jiang',
    author_email='dongfu.jdf@gmail.com',
    packages=find_packages(),
    url='https://yuchenlin.xyz/LLM-Blender/',
    install_requires=[
        'transformers',
        'torch',
        'numpy',
        'accelerate',
        'safetensors',
        'dataclasses-json',
        'sentencepiece',
        'protobuf',
    ],
    extras_require={
        "example": [
            'datasets',
            'scipy',
            'jupyter'
        ],
        "train": [
            'datasets',
            'bitsandbytes',
            'deepspeed',
            'wandb',
        ],
        "data": [
            'datasets',
            'openai',
            'peft',
            'fschat',
        ],
        "eval": [
            'datasets',
            'pycocoevalcap',
            'spacy',
            'prettytable',
            'BLEURT @ git+https://github.com/google-research/bleurt.git@cebe7e6f996b40910cfaa520a63db47807e3bf5c',
            'evaluate',
            'bert_score',
            'tabulate',
            'scipy',
            'nltk',
            'scikit-learn',
            'sacrebleu',
            'rouge_score',
        ],
    },
)
