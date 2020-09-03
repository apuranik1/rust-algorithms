// Run gibbs sampling on an undirected graph
use std::collections::{HashMap, HashSet};

use crate::inference::discrete::{self, DiscreteUndirectedGraph, EdgePotential, NodePotential};

pub struct GraphAssignment<'a, NP: NodePotential, EP: EdgePotential> {
    pub graph: &'a DiscreteUndirectedGraph<NP, EP>,
    pub assignments: Vec<usize>,
}

impl<'a, NP: NodePotential, EP: EdgePotential> GraphAssignment<'a, NP, EP> {
    pub fn new(graph: &'a DiscreteUndirectedGraph<NP, EP>, assignments: Vec<usize>) -> Self {
        GraphAssignment { graph, assignments }
    }
}

impl<'a, NP: NodePotential + Clone, EP: EdgePotential + Clone> GraphAssignment<'a, NP, EP> {
    pub fn resample_node(&self, v: usize) -> usize {
        let neighbors = self.graph.neighbors(v);
        let original = self.graph.node_potential(v);
        let posterior = neighbors
            .iter()
            .fold(original.clone(), |potential, &neighbor| {
                let neighbor_val = self.assignments[neighbor];
                let ep = self.graph.edge_potential(v, neighbor).unwrap();
                let update: Vec<f64> = (0..original.n_values())
                    .map(|node_val| ep.potential(node_val, neighbor_val))
                    .collect();
                potential.update_potentials(&update)
            })
            .to_distribution();
        discrete::sample(&posterior)
    }

    pub fn resample_node_in_place(&mut self, v: usize) {
        self.assignments[v] = self.resample_node(v);
    }

    fn resample_with_labels(&self, nodes: &[usize]) -> Option<(Vec<usize>, Vec<usize>)> {
        let to_resample: HashSet<usize> = nodes.iter().cloned().collect();
        let value_map: HashMap<usize, usize> = (0..self.graph.n_nodes())
            .filter(|n| !to_resample.contains(n))
            .map(|node| (node, self.assignments[node]))
            .collect();
        let (conditioned, new_to_old) = self.graph.condition(value_map);
        conditioned.sample_joint().map(|joint| (joint, new_to_old))
    }

    pub fn resample_block(&self, nodes: &[usize]) -> Option<Vec<usize>> {
        self.resample_with_labels(nodes).map(|(joint, new_to_old)| {
            let old_to_new: HashMap<usize, usize> = new_to_old
                .into_iter()
                .enumerate()
                .map(|(a, b)| (b, a))
                .collect();
            nodes
                .iter()
                .map(|old_idx| joint[old_to_new[old_idx]])
                .collect()
        })
    }

    /// Resample the specified nodes in place
    /// If the nodes do not form a forest, returns false, else true
    pub fn resample_block_in_place(&mut self, nodes: &[usize]) -> bool {
        match self.resample_with_labels(nodes) {
            Some((joint, new_to_old)) => {
                for (value, index) in joint.into_iter().zip(new_to_old.into_iter()) {
                    self.assignments[index] = value;
                }
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::ising::{IsingEdge, IsingNode};
    fn wedge_graph() -> DiscreteUndirectedGraph<IsingNode, IsingEdge> {
        let node_potentials = vec![
            IsingNode::new(0.0),
            IsingNode::new(0.0),
            IsingNode::new(0.0),
        ];
        let mut edge_potentials = HashMap::new();
        edge_potentials.insert((0, 1), IsingEdge::new(10.0));
        edge_potentials.insert((1, 2), IsingEdge::new(-10.0));
        DiscreteUndirectedGraph::new(node_potentials, edge_potentials)
    }

    #[test]
    fn test_resample_block() {
        let wedge = wedge_graph();
        let wedge1 = GraphAssignment::new(&wedge, vec![0, 0, 1]);
        for _ in 0..10 {
            let sample_0 = wedge1.resample_node(0);
            assert_eq!(sample_0, 0);
            let sample_1 = wedge1.resample_node(1);
            assert_eq!(sample_1, 0);
            let sample_2 = wedge1.resample_node(2);
            assert_eq!(sample_2, 1);
        }
        for _ in 0..10 {
            let sample_block = wedge1.resample_block(&[0, 2]).unwrap();
            assert_eq!(1 - sample_block[0], sample_block[1]);
        }
        let mut wedge2 = GraphAssignment::new(&wedge, vec![0, 0, 1]);
        for _ in 0..10 {
            let result = wedge2.resample_block_in_place(&[0, 2]);
            assert!(result);
            assert_eq!(1 - wedge2.assignments[0], wedge2.assignments[2]);
        }
    }

    #[test]
    fn test_resample_block_is_stochastic() {
        let wedge = wedge_graph();
        let wedge_assign = GraphAssignment::new(&wedge, vec![0, 0, 0]);
        let mut n_zeros = vec![0, 0, 0];
        let mut n_ones = vec![0, 0, 0];
        for _ in 0..40 {
            let sample_vals = wedge_assign.resample_block(&[0, 1, 2]).unwrap();
            for (idx, val) in sample_vals.into_iter().enumerate() {
                if val == 0 {
                    n_zeros[idx] += 1;
                } else {
                    n_ones[idx] += 1;
                }
            }
        }
        for &val in n_zeros.iter() {
            assert_ne!(val, 0);
        }
        for &val in n_ones.iter() {
            assert_ne!(val, 0);
        }
    }
}
