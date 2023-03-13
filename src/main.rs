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

fn add_layer(n: &mut NeuralNetwork, inputs: i64, out: i64, r: bool) {
	let mut ne = vec![];
	for _ in 0..inputs {
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

fn main() {

	let mut p = init_neuron(3, true);

	let mut ve = vec![];
	for i in 1..4 {
		let v = value::Value {
			data: i as f64,
			local_grad: 0.0,
			global_grad: 0.0,
			first_child: None,
			second_child: None,
		};
		ve.push(Rc::new(RefCell::new(v)));
	}

	let mut val = activate_neuron(&mut p, &mut ve);
	println!("Value is {}", val.data);
	println!("\n");
	val.local_grad = 1.0;
	value::backward(&mut val, 1.0);

	for i in 0..p.w.len() {
		let w = &p.w[i];
		let mut m = w.borrow_mut();
		println!("Neuron with weight {} and gradient {}", m.data, m.global_grad);
	}

}
