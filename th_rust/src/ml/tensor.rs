#[derive(Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<u32>,
    pub name: String
}

impl Tensor {
    pub fn get_index(&self, crds: Vec<u32>) -> usize {
        if crds.len() != self.shape.len(){
            panic!("Tensor::get_index: missing dimenstion in coordinate given");
        }

        let mut i: i32 = (crds.len()-1) as i32;
        let mut dim_size = 1;
        let mut index: usize = 0;

        while i >= 0 {
            if crds[i as usize] >= self.shape[i as usize] {
                panic!("Tensor::get_index: dim i, invalid value!");
            }

            index = index + (crds[i as usize]*dim_size) as usize;
            dim_size = dim_size*self.shape[i as usize];
            i = i-1;
        }

        return index;
    }

    pub fn lp_norm(&self, p: i32) -> f32 {
        return f32::powf(self.data
                            .iter()
                            .map(|x| f32::powi(*x, p))
                            .sum(), 1.0/(p as f32));
    }

    pub fn forbenius_norm(&self) -> f32 {
        return f32::powf(self.data
                            .iter()
                            .map(|x| f32::powi(*x, 2))
                            .sum(), 0.5);
    }
}

pub fn dot(x: &Tensor, y: &Tensor) -> f32 {
    if x.shape != y.shape {
        panic!("dot product: shape mismatch");
    }

    let product: Vec<f32> = x.data.iter().zip(y.data.iter()).map(|(&a, &b)| a*b).collect();
    let sum: f32 = product.iter().map(|&a| a as f32).sum();
    return sum;
}

pub fn matmul2d(x: &Tensor, y: &Tensor) -> Tensor {
    if x.shape[1] != y.shape[0] {
        panic!("_2dmatmul: incompatible matrices");
    }

    let new_shape = vec![x.shape[0], y.shape[1]];
    let size = new_shape.iter().sum::<u32>();
    let new_data: Vec<f32> = vec![0.0; size as usize];

    let mut z = Tensor{
        data: new_data,
        shape: new_shape,
        name: String::from("mult_ouput")
    };
    
    for i in 0..x.shape[0] {
        for j in 0..y.shape[1] {
            for k in 0..x.shape[1]{
                let z_index = z.get_index(vec![i, j]);
                let x_index = x.get_index(vec![i, k]);
                let y_index = y.get_index(vec![k, j]);
                z.data[z_index] += x.data[x_index]*y.data[y_index];
            }
        }
    }

    return z;
}