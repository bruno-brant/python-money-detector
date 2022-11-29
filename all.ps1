 $env:PYTHONPATH=$PWD
 
 python .\scripts\train_fasterrcnn_resnet50_fpn_v2_untrained.py --dataset-path 'D:\OneDrive\Source\mba\money-dataset\coins_dataset\coins.json' --model-state-dir '.\model_state'
 python .\scripts\train_fasterrcnn_resnet50_fpn_pretrained.py --dataset-path 'D:\OneDrive\Source\mba\money-dataset\coins_dataset\coins.json' --model-state-dir '.\model_state'
 python .\scripts\train_fasterrcnn_resnet50_fpn_v2_pretrained.py --dataset-path 'D:\OneDrive\Source\mba\money-dataset\coins_dataset\coins.json' --model-state-dir '.\model_state'
 python .\scripts\train_fasterrcnn_resnet50_fpn_untrained.py --dataset-path 'D:\OneDrive\Source\mba\money-dataset\coins_dataset\coins.json' --model-state-dir '.\model_state'
