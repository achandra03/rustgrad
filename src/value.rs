use std::cell::RefCell;
use std::rc::Rc;

pub struct Value<'a> {
	pub data:f64,
	pub local_grad:f64,
	pub global_grad:f64,
	pub first_child:Option<Rc<RefCell<Value<'a>>>>,
	pub second_child:Option<Rc<RefCell<Value<'a>>>>
}


pub fn add<'a>(this: Rc<RefCell<Value<'a>>>, other: Rc<RefCell<Value<'a>>>) -> Value<'a> { //a + b
	let data_1 = this.borrow().data;
	let data_2 = other.borrow().data;
	this.borrow_mut().local_grad += 1.0;
	other.borrow_mut().local_grad += 1.0;
	let parent = Value {
		data: data_1 + data_2,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: Some(this),
		second_child: Some(other)
	};
	parent
}


pub fn mult<'a>(this: Rc<RefCell<Value<'a>>>, other: Rc<RefCell<Value<'a>>>) -> Value<'a> { //a * b
	let data_1 = this.borrow().data;
	let data_2 = other.borrow().data;
	this.borrow_mut().local_grad += data_2;
	other.borrow_mut().local_grad += data_1;
	let parent = Value {
		data: data_1 * data_2,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: Some(this),
		second_child: Some(other)
	};
	parent
}



pub fn relu<'a>(this: Rc<RefCell<Value<'a>>>) -> Value<'a> {
	let mut data_1 = this.borrow().data;
	let mut grad:f64 = 1.0;
	if data_1 < 0.0 {
		data_1 = 0.0;
		grad = 0.0;
	}
	this.borrow_mut().local_grad = grad;
	let parent = Value {
		data: data_1,
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: Some(this),
		second_child: None
	};
	parent
	
}

pub fn tanh<'a>(this: Rc<RefCell<Value<'a>>>) -> Value<'a> {
	let data_1 = this.borrow().data;
	this.borrow_mut().local_grad = 1.0 - (data_1.tanh() * data_1.tanh());
	let parent = Value {
		data: data_1.tanh(),
		local_grad: 0.0,
		global_grad: 0.0,
		first_child: Some(this),
		second_child: None
	};
	parent
}


/*
pub fn backward(this: &mut Value, prev_grad: f64) {
	this.global_grad = this.local_grad * prev_grad;

	let first_child = &mut this.first_child;
	match first_child {
		Some(x) => backward(&mut *x, this.global_grad),
		None => ()
	}

	let second_child = &mut this.second_child;
	match second_child {
		Some(x) => backward(&mut *x, this.global_grad),
		None => ()
	}
}

pub fn print_graph(this: &Value) {
	println!("Val: {}, local_grad: {}, global_grad: {}", this.data, this.local_grad, this.global_grad);

	let first_child = &this.first_child;
	match first_child {
		Some(x) => print_graph(&*x),
		None => ()
	}

	let second_child = &this.second_child;
	match second_child {
		Some(x) => print_graph(&*x),
		None => ()
	}
}
*/


fn main() {

}
