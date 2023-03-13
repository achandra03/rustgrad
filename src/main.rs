struct Value<'a> {
	data:f64,
	local_grad:f64,
	global_grad:f64,
	children:Vec<&'a mut Value<'a>>
}

fn add<'a>(this: &'a mut Value<'a>, other: &'a mut Value<'a>) -> Value<'a> {
	let data_1 = this.data;
	let data_2 = other.data;
	this.local_grad += 1.0;
	other.local_grad += 1.0;
	let parent = Value {
		data: data_1 + data_2,
		local_grad: 0.0,
		global_grad: 0.0,
		children: vec![this, other]
	};
	return parent;
}

fn mult<'a>(this: &'a mut Value<'a>, other: &'a mut Value<'a>) -> Value<'a> {
	let data_1 = this.data;
	let data_2 = other.data;
	this.local_grad += data_2;
	other.local_grad += data_1;
	let parent = Value {
		data: data_1 * data_2,
		local_grad: 0.0,
		global_grad: 0.0,
		children: vec![this, other]
	};
	return parent;
}




fn main() {

	
	let mut one = Value {
		data: 1.0,
		local_grad: 0.0,
		global_grad: 0.0,
		children: vec![],
	};

	let mut two = Value {
		data: 2.0,
		local_grad: 0.0,
		global_grad: 0.0,
		children: vec![],
	};

	let parent = mult(&mut one, &mut two);
}
