pub struct Value {
	pub data:f64,
	pub local_grad:f64,
	pub global_grad:f64,
	pub first_child:Option<Box<Value>>,
	pub second_child:Option<Box<Value>>
}

pub fn add(mut this: Value, mut other: Value) -> Value { //a + b
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

pub fn mult(mut this: Value, mut other: Value) -> Value { //a * b
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

pub fn sub(mut this: Value, mut other: Value) -> Value { //a - b
	let data_1 = this.data;
	let data_2 = other.data;
	this.local_grad = 1.0;
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

pub fn div(mut this: Value, mut other: Value) -> Value { //a / b
	let data_1 = this.data;
	let data_2 = other.data;
	this.local_grad = 1.0 / data_2;
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



pub fn relu(mut this: Value) -> Value {
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

pub fn tanh(mut this: Value) -> Value {
	let data_1 = this.data;
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
	let mut g = mult(e, f);
	g.local_grad = 1.0;

	backward(&mut g, 1.0);
	print_graph(&g);
}
