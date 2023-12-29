#[derive(Debug)]
pub struct Matrix {
    pub data: Vec<f32>,
    pub m: u32,
    pub n: u32,
    pub name: String
}

// operator overloading add, sub
// TODO: support broadcasting
impl std::ops::Add<Matrix> for Matrix{
    type Output = Matrix;
    fn add(self, other: Matrix) -> Matrix {
        let mut output = Matrix{
            data: vec![0.0; (self.n*self.m) as usize],
            m: self.m,
            n: self.n,
            name: String::from("addition")
        };

        for i in 0..self.m {
            for j in 0..self.n {
                let self_index = self.get_index(i, j);
                let other_index = other.get_index(i,j);
                let output_index = output.get_index(i,j);
                
                output.data[output_index] = self.data[self_index] + other.data[other_index];
            }
        }

        return output;
    }
}

impl std::ops::Sub<Matrix> for Matrix{
    type Output = Matrix;
    fn sub(self, other: Matrix) -> Matrix {
        let mut output = Matrix{
            data: vec![0.0; (self.n*self.m) as usize],
            m: self.m,
            n: self.n,
            name: String::from("addition")
        };

        for i in 0..self.m {
            for j in 0..self.n {
                let self_index = self.get_index(i, j);
                let other_index = other.get_index(i,j);
                let output_index = output.get_index(i,j);
                
                output.data[output_index] = self.data[self_index] - other.data[other_index];
            }
        }

        return output;
    }
}

impl Matrix {
    pub fn get_index(&self, x: u32, y:u32) -> usize {
        return (x*self.n + y) as usize;
    }

    pub fn forbenius_norm(&self) -> f32 {
        return f32::powf(self.data
                            .iter()
                            .map(|x| f32::powi(*x, 2))
                            .sum(), 0.5);
    }

    pub fn lp_norm(&self, p: i32) -> f32 {
        return f32::powf(self.data
                            .iter()
                            .map(|x| f32::powi(*x, p))
                            .sum(), 1.0/(p as f32));
    }

    pub fn scalar_product(&mut self, k: f32) {
        self.data = self.data.iter().map(|x| x*k).collect();
    }

    // currenlty uses power iteration algorithm
    pub fn dominant_eigan_value(&self, itr: u32) -> f32 {
        let (d_eigen, _ ) = self.dominant_eigans(itr);
        return d_eigen;
    }

    pub fn dominant_eigans(&self, itr: u32) -> (f32, Matrix) {
        let mut x = Matrix {
            data: vec![1.0; self.n as usize],
            m: self.n,
            n: 1,
            name: String::from("eigen_vector")
        };

        for _i in 0..itr {
            x = matmul2d(&self, &x);
            let f_norm = x.forbenius_norm();

            x.scalar_product(1.0/f_norm);
        }

        let eigan_value = matmul2d(
            &matmul2d(&transpose(&x), &self),
            &x
        );

        return (eigan_value.data[0], x);
    }
    
    pub fn get_eigen_values(&self, itr: u32) -> Vec<f32> {
        let mut eigens = vec![0.0; self.n as usize];
        let mut matrix = Matrix {
            data: self.data.clone(),
            m: self.m,
            n: self.n,
            name: String::from("get eignes")
        };

        for i in 0..self.n {
            let (eigen, x) = matrix.dominant_eigans(itr);
            eigens[i as usize] = eigen;

            let mut lambda_xx_t:Matrix = matmul2d(&x, &transpose(&x));
            lambda_xx_t.scalar_product(eigen);
            matrix = matrix - lambda_xx_t;
        }

        return eigens;
    }

    pub fn trace(&self) -> f32 {
        let mut sum: f32 = 0.0;
        for i in 0..self.m {
            for j in 0..self.n{
                if i==j {
                    let index = self.get_index(i, j);
                    sum += self.data[index];
                }
            }
        }
        return sum;
    }

    // TODO:: compute psuedo-inverse through modified svd

}

pub fn transpose(x: &Matrix) -> Matrix {
    let mut y = Matrix { 
        data: vec![0.0; (x.m*x.n) as usize], 
        m: x.n, 
        n: x.m, 
        name: String ::from("transpose matrix")
    };
    for i in 0..x.m{
        for j in 0..x.n{
            let x_index = x.get_index(i,j);
            let y_index = y.get_index(j, i);
            y.data[y_index] = x.data[x_index];
        }
    }

    return y;
}

pub fn dot(x: &Matrix, y: &Matrix) -> f32 {
    if  vec![x.m, x.n] != vec![y.m, y.n] {
        panic!("dot product: shape mismatch");
    }

    let product: Vec<f32> = x.data.iter().zip(y.data.iter()).map(|(&a, &b)| a*b).collect();
    let sum: f32 = product.iter().map(|&a| a as f32).sum();
    return sum;
}

pub fn matmul2d(x: &Matrix, y: &Matrix) -> Matrix {
    if x.n != y.m {
        panic!("_2dmatmul: incompatible matrices");
    }

    let size = x.m * y.n;
    let new_data: Vec<f32> = vec![0.0; size as usize];

    let mut z = Matrix {
        data: new_data,
        m: x.m,
        n: y.n,
        name: String::from("mult_ouput")
    };
    
    for i in 0..x.m {
        for j in 0..y.n {
            for k in 0..x.n{
                let z_index = z.get_index(i, j);
                let x_index = x.get_index(i, k);
                let y_index = y.get_index(k, j);
                z.data[z_index] += x.data[x_index]*y.data[y_index];
            }
        }
    }

    return z;
}

