mod value;
use rand::Rng;

struct Neuron {
	pub w: Vec<value::Value>,
	pub b: value::Value
}

fn init_neuron(n: &mut Neuron, size: i64) {
	let mut rng = rand::thread_rng();
	let mut v = Vec::new();

	for i in 0..size {
		let val = value::Value {
			data: rng.gen_range(-1.0..1.0),
			local_grad: 0.0, 
			global_grad: 0.0, 
			first_child: None, 
			second_child: None
		};
		v.push(val)
	}

	n.w = v;

	let val = value::Value {
		data: 0.0,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: None,
		second_child: None
	};
	n.b = val;
}

fn main() {
	let a = value::Value{
		data: 3.0,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: None,
		second_child: None
	};

	let b = value::Value{
		data: 4.0,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: None,
		second_child: None
	};

	let c = value::Value{
		data: 5.0,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: None,
		second_child: None
	};

	let d = value::Value{
		data: 6.0,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: None,
		second_child: None
	};

	let e = value::mult(a, b);
	let f = value::add(c, d);
	let mut g = value::mult(e, f);
	g.local_grad = 1.0;

	value::backward(&mut g, 1.0);
	value::print_graph(&g);
}
