use std::collections::VecDeque;

#[derive(Clone)]
struct Value {
	data:f64,
	local_grad:f64,
	global_grad:f64,
	first_child:Option<Box<Value>>,
	second_child:Option<Box<Value>>
}

fn add(mut this: Value, mut other: Value) -> Value { //a + b
	let data_1 = this.data;
	let data_2 = other.data;
	this.local_grad += 1.0;
	other.local_grad += 1.0;
	let parent = Value {
		data: data_1 + data_2,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: Some(Box::new(this)),
		second_child: Some(Box::new(other))
	};
	parent
}

fn mult(mut this: Value, mut other: Value) -> Value { //a * b
	let data_1 = this.data;
	let data_2 = other.data;
	this.local_grad += data_2;
	other.local_grad += data_1;
	let parent = Value {
		data: data_1 * data_2,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: Some(Box::new(this)),
		second_child: Some(Box::new(other))
	};
	parent
}

fn sub(mut this: Value, mut other: Value) -> Value { //a - b
	let data_1 = this.data;
	let data_2 = other.data;
	this.local_grad += 1.0;
	other.local_grad -= 1.0;
	let parent = Value {
		data: data_1 - data_2,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: Some(Box::new(this)),
		second_child: Some(Box::new(other))
	};
	parent
}

fn div(mut this: Value, mut other: Value) -> Value { //a / b
	let data_1 = this.data;
	let data_2 = other.data;
	this.local_grad += 1.0 / data_2;
	other.local_grad += -(data_1 / (data_2 * data_2));
	let parent = Value {
		data: data_1 - data_2,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: Some(Box::new(this)),
		second_child: Some(Box::new(other))
	};
	parent
}



fn relu(mut this: Value) -> Value {
	let mut data_1 = this.data;
	let mut grad:f64 = 1.0;
	if data_1 < 0.0 {
		data_1 = 0.0;
		grad = 0.0;
	}
	this.local_grad = grad;
	let parent = Value {
		data: data_1,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: Some(Box::new(this)),
		second_child: None
	};
	parent
	
}

fn tanh(mut this: Value) -> Value {
	let mut data_1 = this.data;
	this.local_grad = 1.0 - (data_1.tanh() * data_1.tanh());
	let parent = Value {
		data: data_1.tanh(),
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: Some(Box::new(this)),
		second_child: None
	};
	parent
}

fn handle_topo(res: &mut Vec<Value>, q: &mut VecDeque<Value>, val: Value) {
	let first_child = &val.first_child;
	match first_child {
		Some(Box) => q.push_back((**Box).clone()),
		None => ()
	}

	let second_child = &val.second_child;
	match second_child {
		Some(Box) => q.push_back((**Box).clone()),
		None => ()
	}

	res.push(val);
}

fn topo(this: Value) -> Vec<Value> {
	let mut q = VecDeque::from([this]);
	let mut res = vec![];
	while q.len() != 0 {
		let curr = q.pop_front();
		match curr {
			Some(Value) => handle_topo(&mut res, &mut q, Value),
			None => ()
		}
	}
	res
}

fn backward(this: Value) {
	let mut res = topo(this);
	res[0].global_grad = 1.0;
	for i in 0..res.len() {
		let mut curr = &mut res[i];

		let first_child = &mut curr.first_child;
		match first_child {
			Some(Box) => (*Box).global_grad = curr.global_grad * (*Box).local_grad,
			None => ()
		}

		let second_child = &mut curr.second_child;
		match second_child {
			Some(Box) => (*Box).global_grad = curr.global_grad * (*Box).local_grad,
			None => ()
		}
	}
}


fn main() {

	let a = Value{
		data: 3.0,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: None,
		second_child: None
	};

	let b = Value{
		data: 4.0,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: None,
		second_child: None
	};

	let c = Value{
		data: 5.0,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: None,
		second_child: None
	};

	let d = Value{
		data: 6.0,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: None,
		second_child: None
	};

	let e = mult(a, b);
	let f = add(c, d);
	let g = mult(e, f);

	backward(g);
}
