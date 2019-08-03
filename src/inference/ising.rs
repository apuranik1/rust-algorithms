use crate::inference::discrete::{EdgePotential, NodePotential};

#[derive(Clone, Debug)]
pub struct IsingNode {
    bias: f64,
}

impl IsingNode {
    pub fn new(bias: f64) -> Self {
        IsingNode { bias }
    }
}

impl NodePotential for IsingNode {
    fn n_values(&self) -> usize {
        2
    }

    fn potential(&self, value: usize) -> f64 {
        match value {
            0 => -self.bias,
            1 => self.bias,
            _ => panic!("Invalid value: {}", value),
        }
    }

    fn update_potentials(&self, updates: &[f64]) -> Self {
        assert!(updates.len() == 2);
        let difference = updates[1] - updates[0];
        IsingNode::new(self.bias + difference / 2.0)
    }
}

#[derive(Clone, Debug)]
pub struct IsingEdge {
    interaction: f64,
}

impl IsingEdge {
    pub fn new(interaction: f64) -> Self {
        IsingEdge { interaction }
    }
}

impl EdgePotential for IsingEdge {
    fn n_values_1(&self) -> usize {
        2
    }

    fn n_values_2(&self) -> usize {
        2
    }

    fn potential(&self, v1: usize, v2: usize) -> f64 {
        if v1 > 1 || v2 > 1 {
            panic!("Invalid values: {}, {}", v1, v2)
        }
        if v1 == v2 {
            self.interaction
        } else {
            -self.interaction
        }
    }

    fn transpose(&self) -> Self {
        self.clone() // Ising potential is symmetrical
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_potential_bias() {
        let np = IsingNode::new(2.0);
        assert_eq!(np.potential(0), -2.0);
        assert_eq!(np.potential(1), 2.0);
    }

    #[test]
    fn test_edge_potential_interaction() {
        let ep = IsingEdge::new(2.0);
        assert_eq!(ep.potential(0, 0), 2.0);
        assert_eq!(ep.potential(0, 1), -2.0);
        assert_eq!(ep.potential(1, 0), -2.0);
        assert_eq!(ep.potential(1, 1), 2.0);
    }

    #[test]
    fn test_potential_update() {
        let np = IsingNode::new(1.0);
        let update = vec![1.0, -1.0];
        assert_eq!(np.update_potentials(&update).bias, 0.0)
    }
}
