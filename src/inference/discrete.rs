use std::collections::BTreeMap;

pub trait NodePotential {
    fn max_value(&self) -> usize;
    fn potential(&self, value: usize) -> f64;
}

pub trait EdgePotential {
    fn max_value_1(&self) -> usize;
    fn max_value_2(&self) -> usize;
    fn potential(&self, v1: usize, v2: usize) -> f64;
}

#[derive(Debug)]
pub struct DiscreteUndirectedGraph<NP: NodePotential, EP: EdgePotential> {
    /// node i takes values between 1 and n_i. phi_i(v) = node_potentials[i][k]
    node_potentials: Vec<NP>,
    /// edges are stored as adjacency matrix, duplicated for undirected edges
    edges: Vec<Vec<usize>>,
    /// psi_ij(v_1, v_2) = edge_potentials[i][j][v_1][v_2]
    edge_potentials: BTreeMap<(usize, usize), EP>,
}

impl<NP: NodePotential, EP: EdgePotential> DiscreteUndirectedGraph<NP, EP> {
    pub fn new(node_potentials: Vec<NP>, edge_potentials: BTreeMap<(usize, usize), EP>) -> Self {
        let n_nodes = node_potentials.len();
        let mut edges: Vec<Vec<usize>> = Vec::new();
        for _ in { 0..n_nodes } {
            edges.push(Vec::new());
        }
        for (v1, v2) in edge_potentials.keys() {
            edges[*v1].push(*v2);
            edges[*v2].push(*v1);
        }
        DiscreteUndirectedGraph {
            node_potentials,
            edges,
            edge_potentials,
        }
    }

    pub fn n_nodes(&self) -> usize {
        self.node_potentials.len()
    }

    pub fn edge_potential(&self, v1: usize, v2: usize) -> Option<&EP> {
        self.edge_potentials
            .get(&(v1, v2))
            .or_else(|| self.edge_potentials.get(&(v2, v1)))
    }

    pub fn node_potential(&self, v: usize) -> &NP {
        &self.node_potentials[v]
    }

    pub fn neighbors(&self, v: usize) -> &Vec<usize> {
        &self.edges[v]
    }

    pub fn degree(&self, v: usize) -> usize {
        self.neighbors(v).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::ising::{IsingEdge, IsingNode};

    fn wedge_graph() -> DiscreteUndirectedGraph<IsingNode, IsingEdge> {
        let node_potentials = vec![
            IsingNode::new(0.0),
            IsingNode::new(1.0),
            IsingNode::new(2.0),
        ];
        let mut edge_potentials = BTreeMap::new();
        edge_potentials.insert((0, 1), IsingEdge::new(0.5));
        edge_potentials.insert((1, 2), IsingEdge::new(-0.5));
        DiscreteUndirectedGraph::new(node_potentials, edge_potentials)
    }

    #[test]
    fn test_correct_edges() {
        let wedge = wedge_graph();
        assert_eq!(*wedge.neighbors(0), vec![1usize]);
        assert_eq!(*wedge.neighbors(2), vec![1usize]);
        let mut neighbors_1 = wedge.neighbors(1).clone();
        neighbors_1.sort();
        assert_eq!(neighbors_1, vec![0usize, 2]);
    }

    #[test]
    fn test_node_potentials() {
        let wedge = wedge_graph();
        assert_eq!(wedge.node_potential(1).potential(0), -1.0);
        assert_eq!(wedge.node_potential(2).potential(1), 2.0);
    }

    #[test]
    fn test_edge_potentials() {
        let wedge = wedge_graph();
        assert_eq!(wedge.edge_potential(0, 1).unwrap().potential(0, 0), 0.5);
        assert!(wedge.edge_potential(0, 2).is_none());
    }
}
