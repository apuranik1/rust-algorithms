use std::collections::BTreeMap;

pub trait NodePotential {
    fn n_values(&self) -> usize;
    fn potential(&self, value: usize) -> f64;
    fn update_potentials(&self, updates: &[f64]) -> Self;
}

pub trait EdgePotential {
    fn n_values_1(&self) -> usize;
    fn n_values_2(&self) -> usize;
    fn potential(&self, v1: usize, v2: usize) -> f64;
    fn transpose(&self) -> Self;
}

#[derive(Debug)]
pub struct DiscreteUndirectedGraph<NP: NodePotential + Clone, EP: EdgePotential + Clone> {
    /// node i takes values between 1 and n_i. phi_i(v) = node_potentials[i][k]
    node_potentials: Vec<NP>,
    /// edges are stored as adjacency matrix, duplicated for undirected edges
    edges: Vec<Vec<usize>>,
    /// psi_ij(v_1, v_2) = edge_potentials[i][j][v_1][v_2]
    edge_potentials: BTreeMap<(usize, usize), EP>,
}

impl<NP: NodePotential + Clone, EP: EdgePotential + Clone> DiscreteUndirectedGraph<NP, EP> {
    pub fn new(
        node_potentials: Vec<NP>,
        mut edge_potentials: BTreeMap<(usize, usize), EP>,
    ) -> Self {
        let n_nodes = node_potentials.len();
        let mut edges: Vec<Vec<usize>> = Vec::new();
        for _ in 0..n_nodes {
            edges.push(Vec::new());
        }
        for (v1, v2) in edge_potentials.keys() {
            edges[*v1].push(*v2);
            if !edge_potentials.contains_key(&(*v2, *v1)) {
                edges[*v2].push(*v1);
            }
        }
        let flipped: Vec<((usize, usize), EP)> = edge_potentials
            .iter()
            .map(|((v1, v2), ep)| ((*v2, *v1), ep.transpose()))
            .collect();
        edge_potentials.extend(flipped);
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
        self.edge_potentials.get(&(v1, v2))
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

    /// Topologically sort the graph from the given root, assuming it is a tree
    /// If the connected component with the root has a cycle, return None
    /// If the graph is not connected, only sort the component with the root
    pub fn tree_topo_sort(&self, tree_root: usize) -> Option<Vec<usize>> {
        let mut topo_order = Vec::new();
        let mut visited = Vec::with_capacity(self.n_nodes());
        for _ in 0..self.node_potentials.len() {
            visited.push(false);
        }
        let mut to_visit = vec![tree_root];

        while !to_visit.is_empty() {
            let next = to_visit.pop().unwrap();
            topo_order.push(next);
            visited[next] = true;
            let mut n_visited_neighbors = 0;
            for neighbor in self.neighbors(next) {
                if visited[*neighbor] {
                    n_visited_neighbors += 1
                } else {
                    to_visit.push(*neighbor)
                }
            }
            if n_visited_neighbors > 1 {
                return None; // cycle detected
            }
        }
        Some(topo_order)
    }

    /// condition the graph on the given values, producing a new graph
    /// returns the new graph and the old labels of the nodes in the new graph
    #[allow(clippy::map_entry)] // false positive with two different maps
    pub fn condition(&self, values: BTreeMap<usize, usize>) -> (Self, Vec<usize>) {
        let new_n_nodes = self.n_nodes() - values.len();
        let mut new_to_old = Vec::with_capacity(new_n_nodes);
        let mut old_to_new = BTreeMap::new();
        // make translations from new to old nodes and vice versa
        for i in 0..self.n_nodes() {
            if !values.contains_key(&i) {
                old_to_new.insert(i, new_to_old.len());
                new_to_old.push(i);
            }
        }
        let mut new_node_potentials = Vec::with_capacity(new_n_nodes);
        let mut new_edge_potentials = BTreeMap::new();
        for (new_index, old_index) in new_to_old.iter().enumerate() {
            let mut new_potential = self.node_potentials[*old_index].clone();
            let n_values = new_potential.n_values();
            // check each old neighbor
            for neighbor in self.neighbors(*old_index) {
                let ep = self.edge_potential(*old_index, *neighbor).unwrap();
                match values.get(neighbor) {
                    Some(&neighbor_val) => {
                        // condition it out
                        let update: Vec<f64> = (0..n_values)
                            .map(|v| ep.potential(v, neighbor_val))
                            .collect();
                        new_potential = new_potential.update_potentials(&update);
                    }
                    None => {
                        // translate to new indices
                        new_edge_potentials.insert((new_index, old_to_new[neighbor]), ep.clone());
                    }
                }
            }
            new_node_potentials.push(new_potential)
        }
        (
            DiscreteUndirectedGraph::new(new_node_potentials, new_edge_potentials),
            new_to_old,
        )
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

    fn square_and_island_graph() -> DiscreteUndirectedGraph<IsingNode, IsingEdge> {
        let node_potentials = vec![
            IsingNode::new(1.0),
            IsingNode::new(1.0),
            IsingNode::new(1.0),
            IsingNode::new(1.0),
            IsingNode::new(1.0),
        ];
        let mut edge_potentials = BTreeMap::new();
        edge_potentials.insert((0, 1), IsingEdge::new(0.5));
        edge_potentials.insert((1, 2), IsingEdge::new(0.5));
        edge_potentials.insert((2, 3), IsingEdge::new(0.5));
        edge_potentials.insert((3, 0), IsingEdge::new(0.5));
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

    #[test]
    fn test_tree_topo_sort() {
        let wedge = wedge_graph();
        let topo_sort = wedge.tree_topo_sort(0);
        assert_eq!(topo_sort, Some(vec![0usize, 1, 2]));
        let topo_sort = wedge.tree_topo_sort(2);
        assert_eq!(topo_sort, Some(vec![2usize, 1, 0]));
    }

    #[test]
    fn test_tree_topo_sort_with_cycle() {
        let graph = square_and_island_graph();
        for start in 0..4 {
            assert!(graph.tree_topo_sort(start).is_none());
        }
        assert_eq!(graph.tree_topo_sort(4), Some(vec![4]));
    }

    #[test]
    fn test_conditioning() {
        let graph = wedge_graph();
        let observations = vec![(0usize, 1usize)]
            .into_iter()
            .collect::<BTreeMap<usize, usize>>();
        let (conditioned, new_to_old) = graph.condition(observations);
        assert_eq!(new_to_old[0], 1);
        assert_eq!(new_to_old[1], 2);
        assert_eq!(conditioned.n_nodes(), 2);
        assert_eq!(conditioned.node_potential(0).potential(1), 1.5);
        assert_eq!(
            conditioned.edge_potential(0, 1).unwrap().potential(0, 0),
            -0.5
        );
        assert_eq!(conditioned.node_potential(1).potential(1), 2.0);
    }
}
