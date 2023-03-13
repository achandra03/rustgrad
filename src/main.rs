mod value;

use rand::Rng;
use std::cell::RefCell;
use std::rc::Rc;

struct Neuron {
	pub w: Vec<Rc<RefCell<value::Value>>>,
	pub b: Rc<RefCell<value::Value>>,
	pub relu: bool
}

struct Layer {
	pub neurons: Vec<Neuron>,
	pub outputs: i64
}

struct NeuralNetwork {
	pub layers: Vec<Layer>
}

fn create_val(x: f64) -> Rc<RefCell<value::Value>> {
	let v = value::Value {
		data: x,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: None,
		second_child: None
	};
	let val = Rc::new(RefCell::new(v));
	val
}

fn init_neuron(size: i64, r: bool) -> Neuron {
	let mut rng = rand::thread_rng();
	let mut v = Vec::new();

	for _ in 0..size {
		let val = value::Value {
			data: rng.gen_range(-1.0..1.0),
			local_grad: 0.0, 
			global_grad: 0.0, 
			first_child: None, 
			second_child: None
		};
		v.push(Rc::new(RefCell::new(val)));
	}

	let val = value::Value {
		data: 0.0,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: None,
		second_child: None
	};

	let n = Neuron {
		w: v,
		b: Rc::new(RefCell::new(val)),
		relu: r
	};

	n
}

fn activate_neuron(n: &mut Neuron, x: &mut Vec<Rc<RefCell<value::Value>>>) -> value::Value {
	
	if x.len() != n.w.len() {
		println!("ERROR: data length {} doesn't equal weight length {}", x.len(), n.w.len());
		std::process::exit(1);
	}

	let mut v: Vec<value::Value> = vec![];
	let len = x.len();
	for i in 0..len {
		let one = Rc::clone(&n.w[i]);
		let two = Rc::clone(&x[i]);
		let val = value::mult(one, two);
		v.push(val);
	}
	
			
	while v.len() != 1 {
		let mut nv = vec![];
		while v.len() >= 2 {
			let one = v.remove(0);
			let two = v.remove(0);
			let val = value::add(Rc::new(RefCell::new(one)), Rc::new(RefCell::new(two)));
			nv.push(val);
		}
		if v.len() == 1 {
			nv.push(v.remove(0));
		}
		v = nv;
	}

	let val =  value::add(Rc::clone(&n.b), Rc::new(RefCell::new(v.remove(0))));

	if n.relu {
		value::relu(Rc::new(RefCell::new(val)))
	} else {
		value::tanh(Rc::new(RefCell::new(val)))
	}
}

fn add_layer(n: &mut NeuralNetwork, neurons: i64, inputs: i64, out: i64, r: bool) { 
	let mut ne = vec![];
	for _ in 0..neurons {
		ne.push(init_neuron(inputs, r));
	}

	let l = Layer {
		neurons: ne, 
		outputs: out
	};

	n.layers.push(l);
}


fn activate_layer(l: &mut Layer, x: &mut Vec<Vec<Rc<RefCell<value::Value>>>>) -> Vec<Vec<Rc<RefCell<value::Value>>>> {
	if x.len() != l.neurons.len() {
		println!("ERROR: output from previous layer length {} doesn't match number of neurons {}", x.len(), l.neurons.len());
		std::process::exit(1);
	}

	let mut res = vec![];
	for _ in 0..l.outputs {
		res.push(vec![]);
	}
	for i in 0..x.len() {
		let xi = &mut x[i];
		let cn = &mut l.neurons[i];
		let output = Rc::new(RefCell::new(activate_neuron(cn, xi)));

		for j in 0..l.outputs {
			res[j as usize].push(Rc::clone(&output));
		}
	}

	res
}


fn forward(net: &mut NeuralNetwork, x: &mut Vec<Vec<Rc<RefCell<value::Value>>>>) -> Vec<Vec<Rc<RefCell<value::Value>>>> {
	let mut curr = x;
	let mut output = vec![];
	for i in 0..net.layers.len() {
		output = activate_layer(&mut net.layers[i], curr);
		if i == net.layers.len() - 1 {
			return output;
		}
		curr = &mut output;
	}

	let res = vec![];
	res
}

fn mse(y_true: &mut Vec<Rc<RefCell<value::Value>>>, y_pred: &mut Vec<Rc<RefCell<value::Value>>>) -> value::Value {
	if y_pred.len() != y_true.len() {
		println!("ERROR: y_pred length {} doesn't equal y_true length {}", y_pred.len(), y_true.len());
		std::process::exit(1);
	}

	let mut diffs = vec![];
	let len = y_pred.len();
	for i in 0..len {
		let one = Rc::clone(&y_true[i]);
		let two = Rc::clone(&y_pred[i]);
		let v = value::sub(one, two);
		let val = value::pow(Rc::new(RefCell::new(v)), 2.0);
		diffs.push(val)
	}

	while diffs.len() != 1 {
		let mut d = vec![];
		while diffs.len() >= 2 {
			let one = diffs.remove(0);
			let two = diffs.remove(0);
			let sum = value::add(Rc::new(RefCell::new(one)), Rc::new(RefCell::new(two)));
			d.push(sum);
		}
		if diffs.len() == 1 {
			d.push(diffs.remove(0));
		}
		diffs = d;
	}

	let denom = value::Value {
		data: len as f64,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: None,
		second_child: None
	};

	let res = value::div(Rc::new(RefCell::new(diffs.remove(0))), Rc::new(RefCell::new(denom)));
	res
}


fn gradient_descent(net: &mut NeuralNetwork, lr: f64, y_true: &mut Vec<Rc<RefCell<value::Value>>>, y_pred: &mut Vec<Rc<RefCell<value::Value>>>) { //step of descent and zeros gradients
	let mut error = mse(y_true, y_pred);
	error.local_grad = 1.0;
	println!("mse is {}", error.data);
	value::backward(&mut error, 1.0);
	let layerlen = net.layers.len();
	for i in 0..layerlen {
		let mut layer = &mut net.layers[i];
		let neuronlen = layer.neurons.len();
		for j in 0..neuronlen {
			let neuron = &mut layer.neurons[j];
			let weightlen = neuron.w.len();
			for k in 0..weightlen {
				let weight = Rc::clone(&neuron.w[k]);
				let mut weight_mut = weight.borrow_mut();
				weight_mut.data -= lr * weight_mut.global_grad;
				weight_mut.global_grad = 0.0;
				weight_mut.local_grad = 0.0;
			}
			let bias = Rc::clone(&neuron.b);
			let mut bias_mut = bias.borrow_mut();
			bias_mut.data -= lr * bias_mut.global_grad;
			bias_mut.local_grad = 0.0;
			bias_mut.global_grad = 0.0;
		}
	}
}

fn main() {

	let mut net = NeuralNetwork {
		layers:vec![]
	};

	add_layer(&mut net, 3, 1, 4, false); //3 neurons each with 1 input and 4 outputs
	add_layer(&mut net, 4, 3, 4, false); //4 neurons each with 3 inputs and 4 output
	add_layer(&mut net, 4, 4, 1, false); //4 neurons each with 4 inputs and 1 output
	add_layer(&mut net, 1, 4, 1, false); //1 neuron each with 4 inputs and 1 output
	
	let mut x = vec![];
	let mut y_true = vec![];

	let mut x_one = vec![];
	x_one.push(create_val(2.0));
	x_one.push(create_val(3.0));
	x_one.push(create_val(-1.0));
	x.push(x_one);

	let mut x_two = vec![];
	x_two.push(create_val(3.0));
	x_two.push(create_val(-1.0));
	x_two.push(create_val(0.5));
	x.push(x_two);

	let mut x_three = vec![];
	x_three.push(create_val(0.5));
	x_three.push(create_val(1.0));
	x_three.push(create_val(1.0));
	x.push(x_three);

	let mut x_four = vec![];
	x_four.push(create_val(1.0));
	x_four.push(create_val(1.0));
	x_four.push(create_val(-1.0));
	x.push(x_four);

	y_true.push(create_val(1.0));
	y_true.push(create_val(-1.0));
	y_true.push(create_val(-1.0));
	y_true.push(create_val(1.0));

	for _ in 0..10 {
		let mut y_pred: Vec<Rc<RefCell<value::Value>>> = vec![];
		let xlen = x.len();
		for i in 0..xlen {
			let mut x_inp = vec![];
			let mut xvec = x[i];
			x_inp.push(xvec);
			let mut forward_pass = forward(&mut net, &mut x_inp);
			let mut out_value = forward_pass.remove(0).remove(0);
			y_pred.push(out_value);
		}

		gradient_descent(&mut net, 0.1, &mut y_true, &mut y_pred);
	}




}
