mod nn;

fn main() {
	
	//Usage example
	let mut net = nn::NeuralNetwork {
		layers:vec![]
	};

	nn::add_layer(&mut net, 3, 1, 4, "tanh".to_string()); //3 neurons each with 1 input and 4 outputs
	nn::add_layer(&mut net, 4, 3, 4, "tanh".to_string()); //4 neurons each with 3 inputs and 4 outputs
	nn::add_layer(&mut net, 4, 4, 1, "tanh".to_string()); //4 neurons each with 4 inputs and 1 output
	nn::add_layer(&mut net, 1, 4, 1, "tanh".to_string()); //1 neuron each with 4 inputs and 1 output

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

	let lr = 0.1;
	for _ in 0..30 {
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

		nn::gradient_descent(&mut net, lr, &mut y_true, &mut y_pred);
	}


		
}
