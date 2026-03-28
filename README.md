# go_neural_net
neural network for graphene oxide based memristors 
Graphene Oxide Analog In-Memory Computing
This project simulates analog neural network inference using real measured conductance states from photoreduced graphene oxide (GO) thin film devices. The goal is to show that GO can serve as a viable analog memory material for energy-efficient AI hardware.
The fabrication and characterization work was done as part of NANO 120A/B at UC San Diego. The ML simulation framework was developed independently.


## ℹ️ what this is
Standard digital neural networks store weights as floating point numbers with essentially infinite precision. Real analog memory devices can only hold a finite number of discrete conductance states. This project asks: how much accuracy do you actually lose when you constrain a neural network to use only the conductance states your physical device can hold?
The short answer, for GO devices: not much.

## 📈📉 key results 
	∙	MNIST: 98.1% accuracy with GO-quantized weights vs 98.2% floating point, a quantization cost of only 0.1%
	∙	State resolution sweep: 7 analog states is sufficient to reach 97.8% accuracy, well within 0.4% of floating point
	∙	CIFAR-10: in progress
	∙	Retention drift: modeled as exponential relaxation toward insulating state, experimental validation ongoing in 120B

## 🔎 how it works 

### 📲 the device 
GO thin films were fabricated on ITO wafers and nylon paper via spin coating and vacuum filtration. Conductance is tuned continuously from insulating to conductive using UV exposure, with each exposure interval corresponding to one programmable memory state. IV curves were measured using a Keithley SMU across 68 device configurations.

### 💻 simulation side 
Conductance values extracted from real IV curve data are normalized and used as the only allowed weight values in a neural network. A custom GOLinear layer replaces standard nn.Linear and uses a straight-through estimator so gradients can flow during training even though weights are quantized. Cycle-to-cycle device noise and retention drift are also modeled.

## 📝 what i've learned 
The biggest challenge was getting CIFAR-10 to work. The first few attempts failed because quantizing the classifier head was killing gradient signal before it could reach the conv layers, so the conv layers never learned anything useful. The fix was two-phase training: first train a full floating point network to get good conv features, then warm-start the GO-quantized head from the float classifier weights and fine-tune. That combination finally got the network learning.
The numpy version of this simulation produced only 45.5% MNIST accuracy because hard weight snapping after every gradient step was preventing the network from learning. Switching to PyTorch with a straight-through estimator brought that up to 98.1%. The STE is the key technique that makes quantization-aware training actually work.

## 📚 stack
Python, PyTorch, NumPy, Matplotlib, torchvision
Hardware: Apple Silicon MPS for GPU acceleration

## 💈 status 
in progress 
