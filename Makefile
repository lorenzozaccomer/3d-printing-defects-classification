
LEARNING_RATE 			= 0.05
EPOCHS 					= 25
MODEL_PATH 				= "./generated_model.pth"
DATASET_PATH 			= "./Images/3d/"
ITERATIONS 				= 0
VISUALIZE_PREDICTION 	= 0

all: clean compile

compile: 
	python3 ./Project/classification.py --model_path $(MODEL_PATH) --dataset_path $(DATASET_PATH) \
	 --iteration $(ITERATIONS) --visualize_prediction $(VISUALIZE_PREDICTION) --learning_rate $(LEARNING_RATE) --epochs $(EPOCHS)

clean:
	rm -f *.txt

delete_model:
	rm -f *.pth