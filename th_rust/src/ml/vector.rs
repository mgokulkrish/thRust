#[derive(Debug)]
pub struct Vector {
    pub data: Vec<f32>,
    pub name: String
}

impl Vector {
    pub fn lp_norm(&self, p:i32) -> f32 {
        return f32::powf(self.data
                            .iter()
                            .map(|x| f32::powi(*x, p))
                            .sum(), 1.0/(p as f32));
    }

    pub fn euclid_norm(&self) -> f32 {
        return self.lp_norm(2);
    }

    pub fn max_norm(&self) -> f32 {
        match self.data.iter()
                        .cloned()
                        .max_by(|a, b| a.partial_cmp(b).unwrap()) {
            Some(max) => max,
            None => panic!("Vector::max_norm: norm of an empty vector!!")
        }
    }
}