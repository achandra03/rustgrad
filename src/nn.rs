pub mod value;

use rand::Rng;
use std::cell::RefCell;
use std::rc::Rc;
use std::collections::HashMap;

pub struct Neuron {
	pub w: Vec<Rc<RefCell<value::Value>>>,
	pub b: Rc<RefCell<value::Value>>,
	pub act: String
}

pub struct Layer {
	pub neurons: Vec<Neuron>,
	pub outputs: i64
}

pub struct NeuralNetwork {
	pub layers: Vec<Layer>
}

pub fn create_val(x: f64) -> Rc<RefCell<value::Value>> {
	let v = value::Value {
		data: x,
		local_grads: HashMap::new(),
		global_grad: 0.0,
		first_child: None,
		second_child: None,
		id: value::random_string(50)
	};
	let val = Rc::new(RefCell::new(v));
	val
}

fn init_neuron(size: i64, ac: String) -> Neuron {
	let mut rng = rand::thread_rng();
	let mut v = Vec::new();

	for _ in 0..size {
		let val = create_val(rng.gen_range(-1.0..1.0));
		v.push(val);
	}
	let val = create_val(0.0);

	let n = Neuron {
		w: v,
		b: val,
		act: ac
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

	if n.act.eq("relu") {
		return value::relu(Rc::new(RefCell::new(val)));
	} else if n.act.eq("tanh") {
		return value::tanh(Rc::new(RefCell::new(val)));
	} 

	val
}

pub fn add_layer(n: &mut NeuralNetwork, neurons: i64, inputs: i64, out: i64, act: String) { 
	let mut ne = vec![];
	for _ in 0..neurons {
		ne.push(init_neuron(inputs, act.clone()));
	}

	if n.layers.len() == 0 { //input weights should not modify data
		for i in 0..neurons {
			let neuron = &mut ne[i as usize];
			let weightlen = neuron.w.len();
			
			for j in 0..weightlen {
				let weight = &mut neuron.w[j];
				weight.borrow_mut().data = 1.0;
			}
		}
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


pub fn forward(net: &mut NeuralNetwork, x: &mut Vec<Vec<Rc<RefCell<value::Value>>>>) -> Vec<Vec<Rc<RefCell<value::Value>>>> {
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

	let denom = create_val(len as f64);

	let res = value::div(Rc::new(RefCell::new(diffs.remove(0))), denom);
	res
}


pub fn gradient_descent(net: &mut NeuralNetwork, lr: f64, y_true: &mut Vec<Rc<RefCell<value::Value>>>, y_pred: &mut Vec<Rc<RefCell<value::Value>>>) { //step of descent and zeros gradients
	let mut error = mse(y_true, y_pred);
	error.global_grad = 1.0;
	println!("mse is {}", error.data);
	value::backward(Rc::new(RefCell::new(error)));
	let layerlen = net.layers.len();
	for i in 1..layerlen {
		let layer = &mut net.layers[i];
		let neuronlen = layer.neurons.len();
		for j in 0..neuronlen {
			let neuron = &mut layer.neurons[j];
			let weightlen = neuron.w.len();
			for k in 0..weightlen {
				let weight = Rc::clone(&neuron.w[k]);
				let mut weight_mut = weight.borrow_mut();
				weight_mut.data -= lr * weight_mut.global_grad;
				weight_mut.global_grad = 0.0;
				weight_mut.local_grads = HashMap::new();
			}
			let bias = Rc::clone(&neuron.b);
			let mut bias_mut = bias.borrow_mut();
			bias_mut.data -= lr * bias_mut.global_grad;
			bias_mut.local_grads = HashMap::new();
			bias_mut.global_grad = 0.0;
		}
	}
}

pub fn print_weights(net: &mut NeuralNetwork) {
	let layerlen = net.layers.len();
	for i in 0..layerlen {
		let layer = &mut net.layers[i];
		let neuronlen = layer.neurons.len();
		for j in 0..neuronlen {
			let neuron = &mut layer.neurons[j];
			let weightlen = neuron.w.len();
			for k in 0..weightlen {
				let weight = Rc::clone(&neuron.w[k]);
				let weight_mut = weight.borrow_mut();
				println!("Layer {} neuron {} weight {}, with value {} and gradient {}", i, j, k, weight_mut.data, weight_mut.global_grad);
			}
			let bias = Rc::clone(&neuron.b);
			let bias_mut = bias.borrow_mut();
			println!("Layer {} neuron {} bias, with value {} and gradient {}", i, j, bias_mut.data, bias_mut.global_grad);
		}
	}
}

