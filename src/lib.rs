#![feature(zero_one)]
#![allow(unused_imports)]

extern crate rand;

pub mod linsys {
	pub mod vector;
	pub mod matrix;
	pub mod dim;
}

pub mod ml {
	pub mod k_means;
	pub mod knn;
	pub mod linear_ls;
}

// pub mod trees {
// 	pub mod kd_tree;
// }

// #[cfg(test)]
// mod tests {
//     #[test]
//     fn it_works() {
//     }
// }