use std::cell::RefCell;
use std::rc::Rc;
use rand::{distributions::Alphanumeric, Rng};
use std::collections::HashMap;

#[derive(Clone)]
pub struct Value {
	pub data:f64,
	pub local_grads: HashMap<String, f64>,
	pub global_grad:f64,
	pub first_child:Option<Rc<RefCell<Value>>>,
	pub second_child:Option<Rc<RefCell<Value>>>,
	pub id: String
}

pub fn random_string(len: usize) -> String {
	let s: String = rand::thread_rng()
		.sample_iter(&Alphanumeric)
		.take(len)
		.map(char::from)
		.collect();
	s	
}


pub fn add(this: Rc<RefCell<Value>>, other: Rc<RefCell<Value>>) -> Value { //a + b
	let data_1 = this.borrow().data;
	let data_2 = other.borrow().data;
	let i = random_string(50);
	this.borrow_mut().local_grads.insert(i.clone(), 1.0);
	other.borrow_mut().local_grads.insert(i.clone(), 1.0);
	let parent = Value {
		data: data_1 + data_2,
		local_grads: HashMap::new(),
		global_grad: 0.0,
		first_child: Some(this),
		second_child: Some(other),
		id: i
	};
	parent
}


pub fn mult(this: Rc<RefCell<Value>>, other: Rc<RefCell<Value>>) -> Value { //a * b
	let data_1 = this.borrow().data;
	let data_2 = other.borrow().data;
	let i = random_string(50);
	this.borrow_mut().local_grads.insert(i.clone(), data_2);
	other.borrow_mut().local_grads.insert(i.clone(), data_1);
	let parent = Value {
		data: data_1 * data_2,
		local_grads: HashMap::new(),
		global_grad: 0.0,
		first_child: Some(this),
		second_child: Some(other),
		id: i
	};
	parent
}

pub fn sub(this: Rc<RefCell<Value>>, other: Rc<RefCell<Value>>) -> Value { //a - b 
	let data_1 = this.borrow().data;
	let data_2 = other.borrow().data;
	let i = random_string(50);
	this.borrow_mut().local_grads.insert(i.clone(), 1.0);
	other.borrow_mut().local_grads.insert(i.clone(), -1.0);
	let parent = Value {
		data: data_1 - data_2,
		local_grads: HashMap::new(),
		global_grad: 0.0,
		first_child: Some(this),
		second_child: Some(other),
		id: i
	};
	parent
}

pub fn div(this: Rc<RefCell<Value>>, other: Rc<RefCell<Value>>) -> Value { //a / b 
	let data_1 = this.borrow().data;
	let data_2 = other.borrow().data;
	let i = random_string(50);
	this.borrow_mut().local_grads.insert(i.clone(), 1.0 / data_2);
	other.borrow_mut().local_grads.insert(i.clone(), -data_1 / (data_2 * data_2));
	let parent = Value {
		data: data_1 / data_2,
		local_grads: HashMap::new(),
		global_grad: 0.0,
		first_child: Some(this),
		second_child: Some(other),
		id: i
	};
	parent
}

pub fn pow(this: Rc<RefCell<Value>>, other: f64) -> Value {
	let data_1 = this.borrow().data;
	let i = random_string(50);
	this.borrow_mut().local_grads.insert(i.clone(), other * data_1.powf(other - 1.0));
	let parent = Value {
		data: data_1.powf(other),
		local_grads: HashMap::new(),
		global_grad: 0.0,
		first_child: Some(this),
		second_child: None,
		id: i
	};
	parent
}



pub fn relu(this: Rc<RefCell<Value>>) -> Value {
	let mut data_1 = this.borrow().data;
	let mut grad:f64 = 1.0;
	if data_1 < 0.0 {
		data_1 = 0.0;
		grad = 0.0;
	}
	let i = random_string(50);
	this.borrow_mut().local_grads.insert(i.clone(), grad);
	let parent = Value {
		data: data_1,
		local_grads: HashMap::new(),
		global_grad: 0.0,
		first_child: Some(this),
		second_child: None,
		id: i
	};
	parent
	
}

pub fn tanh(this: Rc<RefCell<Value>>) -> Value {
	let data_1 = this.borrow().data;
	let i = random_string(50);
	this.borrow_mut().local_grads.insert(i.clone(), 1.0 - (data_1.tanh() * data_1.tanh()));
	let parent = Value {
		data: data_1.tanh(),
		local_grads: HashMap::new(),
		global_grad: 0.0,
		first_child: Some(this),
		second_child: None,
		id: i
	};
	parent
}


pub fn backward(this: &mut Value, prev_grad: f64, prev_id: String) {
	if prev_id.ne("") {
		let lg = this.local_grads.get(&prev_id);
		let mut local_grad = 0.0;
		match lg {
			Some(d) => local_grad = *d,
			None => println!("Not found")
		};

		this.global_grad += local_grad * prev_grad;
	}

	let first_child = &mut this.first_child;
	match first_child {
		Some(x) => backward(&mut x.borrow_mut(), this.global_grad, this.id.clone()),
		None => ()
	}

	let second_child = &mut this.second_child;
	match second_child {
		Some(x) => backward(&mut x.borrow_mut(), this.global_grad, this.id.clone()),
		None => ()
	}
}

pub fn print_graph(this: &Value) {
	println!("Val: {}, global_grad: {}", this.data, this.global_grad);

	let first_child = &this.first_child;
	match first_child {
		Some(x) => print_graph(&x.borrow()),
		None => ()
	}

	let second_child = &this.second_child;
	match second_child {
		Some(x) => print_graph(&x.borrow()),
		None => ()
	}
}
