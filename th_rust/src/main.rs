use crate::ml::matrix;
// use crate::ml::tensor;
// use crate::ml::vector;
pub mod ml;


fn main() {
    
    let z = matrix::Matrix{
        data: vec![1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0],
        m: 3,
        n: 3,
        name: String::from("tensor z"),
    };

    println!("dominant eigan value of matrix z is {}", z.dominant_eigan_value(10));
    println!("trace of z : {}", z.trace());

    println!("eigan_values of z: {:?}", z.get_eigen_values(10));

}
