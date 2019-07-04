use num::complex::Complex64;
use num::Zero;

/// Compute the fourier transform of data
/// The order of the transform is data.len() rounded up to a power of 2
pub fn fft(data: &[Complex64]) -> Vec<Complex64> {
    let order = power_of_2_geq_n(data.len());
    eval_polynomial(order, true, data, 0, 1)
}

/// Compute the inverse fourier transform of data
/// The order is data.len(), which must be a power of 2
/// If the order is not a power of 2, returns None
pub fn ifft(data: &[Complex64]) -> Option<Vec<Complex64>> {
    let order = power_of_2_geq_n(data.len());
    if order != data.len() {
        None
    } else {
        let scale_factor = 1.0 / (order as f64);
        Some(
            eval_polynomial(order, false, data, 0, 1)
                .iter()
                .map(|v: &Complex64| v.scale(scale_factor))
                .collect(),
        )
    }
}

fn power_of_2_geq_n(n: usize) -> usize {
    match n {
        0 => 1,
        1 => 1,
        _ => {
            if n % 2 == 0 {
                2 * power_of_2_geq_n(n / 2)
            } else {
                2 * power_of_2_geq_n(n / 2 + 1) // round up
            }
        }
    }
}

/// Evaluate the polynomial given by a_j = coefs[offset + j * stride] at all the roots
/// of unity of order coefs.len() / stride
/// Assumes order is a power of 2 (WILL DO UNEXPECTED THINGS IF IT ISN'T)
fn eval_polynomial(
    order: usize,
    negative_angle: bool,
    coefs: &[Complex64],
    offset: usize,
    stride: usize,
) -> Vec<Complex64> {
    assert!(offset < stride);
    if order == 0 {
        panic!("Illegal state: coefs.len() < stride")
    } else if order == 1 {
        // If it's off the end, then implicitly zero-pad
        match coefs.get(offset) {
            Some(c) => vec![*c],
            None => vec![Complex64::zero()],
        }
    } else {
        let even_result = eval_polynomial(order / 2, negative_angle, coefs, offset, stride * 2);
        let odd_result = eval_polynomial(
            order / 2,
            negative_angle,
            coefs,
            offset + stride,
            stride * 2,
        );
        let angle =
            (if negative_angle { -1.0 } else { 1.0 }) * 2.0 * std::f64::consts::PI / (order as f64);
        let mut first_half = Vec::new();
        let mut second_half = Vec::new();
        for (k, (even_part, odd_part)) in
            Iterator::zip(even_result.iter(), odd_result.iter()).enumerate()
        {
            let positive_root = Complex64::from_polar(&(1.0), &(angle * k as f64));
            // at postive_root, P = even_part + positive_root * odd_part
            first_half.push(even_part + positive_root * odd_part);
            // at -positive_root, do something similar
            second_half.push(even_part - positive_root * odd_part);
        }
        first_half.extend(second_half);
        first_half
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_close<'a, T: Iterator<Item = &'a Complex64>>(first: T, second: T, tol: f64) {
        for (a, b) in Iterator::zip(first, second) {
            assert!((b - a).norm_sqr() < tol);
        }
    }

    #[test]
    fn test_nearest_power_of_2() {
        assert_eq!(power_of_2_geq_n(16), 16);
        assert_eq!(power_of_2_geq_n(17), 32);
        assert_eq!(power_of_2_geq_n(24), 32);
    }

    #[test]
    fn test_fourier_sin() {
        let data = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ];
        let expected_ft = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -2.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 2.0),
        ];
        let fourier = fft(&data);
        assert_all_close(expected_ft.iter(), fourier.iter(), 1e-12);
    }

    #[test]
    fn test_fourier_cos() {
        let data = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ];
        let expected_ft = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
        ];
        let fourier = fft(&data);
        assert_all_close(expected_ft.iter(), fourier.iter(), 1e-12);
    }

    #[test]
    fn test_inverse_fourier_sin() {
        let data = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -2.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 2.0),
        ];
        let expected_ift = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ];
        let inv_fourier = ifft(&data).unwrap();
        assert_all_close(expected_ift.iter(), inv_fourier.iter(), 1e-12);
    }

    #[test]
    fn test_big_fourier_roundtrip() {
        let data: Vec<Complex64> = (0..10000).map(|x| Complex64::new(x as f64, 0.0)).collect();
        let reconstructed = ifft(&fft(&data)).unwrap();
        assert_all_close(data.iter(), reconstructed.iter(), 1e-12);
    }
}
