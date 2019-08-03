use crate::inference::discrete::{EdgePotential, NodePotential};

#[derive(Debug)]
pub struct IsingNode {
    bias: f64,
}

impl IsingNode {
    pub fn new(bias: f64) -> Self {
        IsingNode { bias }
    }
}

impl NodePotential for IsingNode {
    fn max_value(&self) -> usize {
        2
    }

    fn potential(&self, value: usize) -> f64 {
        match value {
            0 => -self.bias,
            1 => self.bias,
            _ => panic!("Invalid value: {}", value),
        }
    }
}

#[derive(Debug)]
pub struct IsingEdge {
    interaction: f64,
}

impl IsingEdge {
    pub fn new(interaction: f64) -> Self {
        IsingEdge { interaction }
    }
}

impl EdgePotential for IsingEdge {
    fn max_value_1(&self) -> usize {
        2
    }

    fn max_value_2(&self) -> usize {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::discrete::DiscreteUndirectedGraph;

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
}
