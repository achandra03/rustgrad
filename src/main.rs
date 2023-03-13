mod value;
use rand::Rng;

struct Neuron {
	pub w: Vec<value::Value>,
	pub b: value::Value,
	pub relu: bool
}

struct Layer {
	pub neurons: Vec<Neuron>,
	pub outputs: i64
}

struct NeuralNetwork {
	pub layers: Vec<Layer>
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
		v.push(val)
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
		b: val,
		relu: r
	};

	n
}

fn activate_neuron(n: &mut Neuron, x: &mut Vec<value::Value>) -> value::Value {
	if x.len() != n.w.len() {
		println!("ERROR: data length {} doesn't equal weight length {}", x.len(), n.w.len());
		std::process::exit(1);
	}

	let mut v = vec![];
	while x.len() != 0 {
		let one = x.remove(0);
		let two = n.w.remove(0);
		let val = value::mult(one, two);
		v.push(val);
	}

	while v.len() != 1 {
		let mut nv = vec![];
		while v.len() >= 2 {
			let one = v.remove(0);
			let two = v.remove(0);
			let val = value::add(one, two);
			nv.push(val);
		}
		if v.len() == 1 {
			nv.push(v.remove(0));
		}
	}

	if n.relu {
		value::relu(v.remove(0))
	} else {
		value::tanh(v.remove(0))
	}
}

fn activate_layer(l: &mut Layer, x: &mut Vec<Vec<value::Value>>) -> Vec<Vec<value::Value>> {
	if x.len() != l.neurons.len() {
		println!("ERROR: ouput from previous layer length {} doesn't match number of neurons {}", x.len(), l.neurons.len());
		std::process::exit(1);
	}

	let mut res = vec![];
	for _ in 0..l.outputs {
		res.push(vec![]);
	}
	for i in 0..x.len() {
		let xi = &mut x[i];
		let cn = &mut l.neurons[i];
		let output = activate_neuron(cn, xi);

		for j in 0..l.outputs {
			res[j as usize].push(output.clone());
		}
	}

	res
}

fn feedforward(net: &mut NeuralNetwork, x: &mut Vec<Vec<value::Value>>) -> Vec<Vec<value::Value>> {
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

fn main() {

}
