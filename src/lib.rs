#![feature(zero_one)]
#![allow(unused_imports)]

extern crate rand;

pub mod linsys {
	pub mod vector;
	pub mod matrix;
}

pub mod ml {
	pub mod k_means;
}

// #[cfg(test)]
// mod tests {
//     #[test]
//     fn it_works() {
//     }
// }