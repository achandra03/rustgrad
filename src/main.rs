mod nn;
use std::cell::RefCell;
use std::rc::Rc;
fn main() {
	
	/*
	let mut net = nn::NeuralNetwork {
		layers:vec![]
	};
	
	nn::add_layer(&mut net, 1, 1, 3, false);
	nn::add_layer(&mut net, 3, 1, 1, false);
	nn::add_layer(&mut net, 1, 3, 1, false);

	let mut x = vec![];
	let mut y_true = vec![];

	for i in 1..10 {
		let mut data = i;
		let mut res = 1;

		if data % 2 == 1 {
			res = -1;
		}

		let mut xxi = vec![];
		xxi.push(nn::nn::create_val(data as f64));
		let mut xi = vec![];
		xi.push(xxi);
		x.push(xi);
		y_true.push(nn::nn::create_val(res as f64));
	}

	for _ in 0..10 {
		let mut y_pred = vec![];
		let xlen = x.len();
		for i in 0..xlen {
			let mut xi = &mut x[i];
			let mut forward_pass = nn::forward(&mut net, xi);
			let mut output = forward_pass.remove(0).remove(0);
			y_pred.push(output);
		}

		nn::gradient_descent(&mut net, 0.1, &mut y_true, &mut y_pred);
	}
	*/


	/*	
	let mut net = nn::NeuralNetwork {
		layers:vec![]
	};

	nn::add_layer(&mut net, 3, 1, 4, false); //3 neurons each with 1 input and 4 outputs
	nn::add_layer(&mut net, 4, 3, 4, false); //4 neurons each with 3 inputs and 4 outputs
	nn::add_layer(&mut net, 4, 4, 1, false); //4 neurons each with 4 inputs and 1 output
	nn::add_layer(&mut net, 1, 4, 1, false); //1 neuron each with 4 inputs and 1 output

	let mut x = vec![];

	let mut x_one = vec![];
	x_one.push(nn::create_val(2.0));
	x_one.push(nn::create_val(3.0));
	x_one.push(nn::create_val(-1.0));
	x.push(x_one);

	let mut x_two = vec![];
	x_two.push(nn::create_val(3.0));
	x_two.push(nn::create_val(-1.0));
	x_two.push(nn::create_val(0.5));
	x.push(x_two);

	let mut x_three = vec![];
	x_three.push(nn::create_val(0.5));
	x_three.push(nn::create_val(1.0));
	x_three.push(nn::create_val(1.0));
	x.push(x_three);

	let mut x_four = vec![];
	x_four.push(nn::create_val(1.0));
	x_four.push(nn::create_val(1.0));
	x_four.push(nn::create_val(-1.0));
	x.push(x_four);
	
	let mut y_true = vec![];
	y_true.push(nn::create_val(1.0));
	y_true.push(nn::create_val(-1.0));
	y_true.push(nn::create_val(-1.0));
	y_true.push(nn::create_val(1.0));

	for _ in 0..50 {
		let mut y_pred = vec![];
		let xlen = x.len();
		for i in 0..xlen {
			let mut xi = vec![];
			for j in 0..3 {
				let mut currx = vec![];
				currx.push(x[i][j].clone());
				xi.push(currx);
			}
			let mut forward_pass = nn::forward(&mut net, &mut xi);
			let out_value = forward_pass.remove(0).remove(0);
			y_pred.push(out_value);
		}

		nn::gradient_descent(&mut net, 0.005, &mut y_true, &mut y_pred);
	}
	*/

	let mut net = nn::NeuralNetwork {
		layers:vec![]
	};
	
//	nn::add_layer(&mut net


}
