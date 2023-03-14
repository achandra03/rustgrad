use std::cell::RefCell;
use std::rc::Rc;
use rand::{distributions::Alphanumeric, Rng};
use std::collections::HashMap;
use std::collections::HashSet;

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
	let data_2 = other.borrow().data;
	if data_2 == 0.0 {
		println!("ERROR: division by 0");
		std::process::exit(1);
	}
	let data_1 = this.borrow().data;
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

pub fn pow(this: Rc<RefCell<Value>>, other: Rc<RefCell<Value>>) -> Value {
	let data_1 = this.borrow().data;
	let data_2 = other.borrow().data;
	let i = random_string(50);
	this.borrow_mut().local_grads.insert(i.clone(), data_2 * data_1.powf(data_2 - 1.0));
	other.borrow_mut().local_grads.insert(i.clone(), data_1.powf(data_2) * data_1.ln());
	let parent = Value {
		data: data_1.powf(data_2),
		local_grads: HashMap::new(),
		global_grad: 0.0,
		first_child: Some(this),
		second_child: Some(other),
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

pub fn sigmoid(this: Rc<RefCell<Value>>) -> Value {
	let data_1 = this.borrow().data;
	let i = random_string(50);
	let sig = 1.0 / (1.0 + (-data_1).exp());
	this.borrow_mut().local_grads.insert(i.clone(), sig * (1.0 - sig));
	let parent = Value {
		data: sig,
		local_grads: HashMap::new(),
		global_grad: 0.0,
		first_child: Some(this),
		second_child: None,
		id: i
	};
	parent
}

pub fn handle_topo(x: Rc<RefCell<Value>>, q: &mut Vec<Rc<RefCell<Value>>>, visited: &mut HashSet<String>) {
	let id = x.borrow().id.clone();
	if !visited.contains(&id) {
		q.push(Rc::clone(&x));
		visited.insert(id);
	}
	
}


pub fn backward(this: Rc<RefCell<Value>>) {
	let mut topo = vec![];
	let mut q = vec![];
	let mut visited = HashSet::new();

	q.push(Rc::clone(&this));
	visited.insert(this.borrow().id.clone());

	while q.len() != 0 {
		let curr = q.remove(0);
		match &mut curr.borrow_mut().first_child {
			Some(x) => handle_topo(Rc::clone(&x), &mut q, &mut visited),
			None => ()
		}

		match &mut curr.borrow_mut().second_child {
			Some(x) => handle_topo(Rc::clone(&x), &mut q, &mut visited),
			None => ()
		}

		topo.push(curr);
	}

	let mut global_grads = HashMap::new();
	let last = topo.remove(0);
	last.borrow_mut().global_grad = 1.0;
	global_grads.insert(last.borrow().id.clone(), 1.0);
	last.borrow_mut().local_grads = HashMap::new();

	while topo.len() != 0 {
		let curr = topo.remove(0);
		let mut sum = 0.0;
		for (id, grad) in &curr.borrow().local_grads {
			match global_grads.get(id) {
				Some(global_grad) => sum += grad * global_grad,
				None => println!("ERROR: corresponding global gradient not found. This shouldn't happen")
			}
		}
		curr.borrow_mut().global_grad = sum;
		global_grads.insert(curr.borrow().id.clone(), sum);
		curr.borrow_mut().local_grads = HashMap::new();
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
