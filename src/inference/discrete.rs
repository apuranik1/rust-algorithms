use std::collections::{BTreeMap, BTreeSet};

// All values are in ln-space unless otherwise specified

pub trait NodePotential {
    fn n_values(&self) -> usize;
    fn potential(&self, value: usize) -> f64;
    fn update_potentials(&self, updates: &[f64]) -> Self;

    fn to_distribution(&self) -> Vec<f64> {
        let unnorm_values: Vec<f64> = (0..self.n_values())
            .map(|x| self.potential(x).exp())
            .collect();
        let norm_const: f64 = unnorm_values.iter().sum();
        unnorm_values.iter().map(|x| x / norm_const).collect()
    }
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

type Message = (usize, Vec<f64>);

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

    pub fn set_node_potential(&mut self, v: usize, potential: NP) {
        self.node_potentials[v] = potential;
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
        let mut to_visit = vec![tree_root];
        // a node is marked as visited when it is first added to to_visit
        let mut visited = BTreeSet::new();
        visited.insert(tree_root);

        while let Some(next) = to_visit.pop() {
            topo_order.push(next);
            let mut n_visited_neighbors = 0;
            for neighbor in self.neighbors(next) {
                if visited.contains(neighbor) {
                    n_visited_neighbors += 1;
                } else {
                    to_visit.push(*neighbor);
                    visited.insert(next);
                }
            }
            if n_visited_neighbors > 1 {
                return None; // cycle detected
            }
        }
        Some(topo_order)
    }

    /// Topologically sort the full graph, assuming it is a forest
    /// roots are always the nodes with lowest index in a tree
    /// If the graph is not a forest, returns None
    pub fn forest_topo_sort(&self) -> Option<Vec<usize>> {
        let mut topo_sort = Vec::with_capacity(self.n_nodes());
        let mut visited = Vec::with_capacity(self.n_nodes());
        for _ in 0..self.node_potentials.len() {
            visited.push(false);
        }
        for candidate_root in 0..self.n_nodes() {
            if !visited[candidate_root] {
                // add a new tree, rooted here
                match self.tree_topo_sort(candidate_root) {
                    Some(mut sort_order) => {
                        for &node in sort_order.iter() {
                            visited[node] = true;
                        }
                        topo_sort.append(&mut sort_order);
                    }
                    None => {
                        return None;
                    }
                }
            }
        }
        Some(topo_sort)
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

    /// Compute the marginal distribution at each node of the graph
    /// Each marginal is expressed as a node potential
    pub fn compute_marginals(&self) -> Option<Vec<NP>> {
        self.forest_topo_sort().map(|node_order| {
            let mut visited: Vec<bool> = Vec::with_capacity(self.n_nodes());
            let mut inboxes = self.make_inboxes();
            for _ in 0..self.n_nodes() {
                visited.push(false);
            }
            // belief propagation up from leaves
            for &node in node_order.iter().rev() {
                // mark it as visited
                visited[node] = true;
                // for each unvisited neighbor: send a message
                for &neighbor in self.neighbors(node).iter() {
                    if !visited[neighbor] {
                        let new_message = self.message(node, neighbor, &inboxes);
                        inboxes[neighbor].push(new_message);
                    }
                }
            }
            // reset visited array
            for entry in visited.iter_mut() {
                *entry = false;
            }
            // belief propagation down from root(s), still accumulating to inbox
            for &node in node_order.iter() {
                visited[node] = true;
                for &neighbor in self.neighbors(node).iter() {
                    if !visited[neighbor] {
                        let new_message = self.message(node, neighbor, &inboxes);
                        inboxes[neighbor].push(new_message);
                    }
                }
            }
            self.node_potentials
                .iter()
                .zip(inboxes.iter())
                .map(|(np, inbox)| {
                    inbox
                        .iter()
                        .fold(np.clone(), |p, msg| p.update_potentials(&msg.1))
                })
                .collect()
        })
    }

    fn make_inboxes(&self) -> Vec<Vec<Message>> {
        let mut inboxes: Vec<Vec<Message>> = Vec::with_capacity(self.n_nodes());
        for _ in 0..self.n_nodes() {
            inboxes.push(Vec::new());
        }
        inboxes
    }

    fn message(&self, node: usize, neighbor: usize, inboxes: &[Vec<Message>]) -> Message {
        // message_i->j(x_j) = sum_{x_i}phi_i(x_i)psi_ij(x_j, x_i) prod_{k not j} m_k->i(x_i)
        let inbound = &inboxes[node];
        let phi_i = self.node_potential(node);
        let phi_j = self.node_potential(neighbor);
        let psi_ij = self.edge_potential(node, neighbor).unwrap();
        let mut outbound = Vec::new();
        for x_j in 0..phi_j.n_values() {
            // compute m_node->neighbor (x_j)
            let mut to_sum = Vec::new();
            for x_i in 0..phi_i.n_values() {
                let base_potential = phi_i.potential(x_i) + psi_ij.potential(x_i, x_j);
                // aggregate messages not from neighbor
                let message_agg: f64 = inbound
                    .iter()
                    .filter_map(|message| {
                        if message.0 == neighbor {
                            None
                        } else {
                            Some(message.1[x_i])
                        }
                    })
                    .sum();
                to_sum.push(base_potential + message_agg);
            }
            outbound.push(log_sum_exp(&to_sum));
        }
        (node, outbound)
    }
}

/// Compute the logsumexp of the values
/// If values is empty, returns NaN
fn log_sum_exp(values: &[f64]) -> f64 {
    use std::f64;
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let sum_exp = values.iter().fold(0.0, |a, &b| a + (b - max_val).exp());
    sum_exp.ln() + max_val
}

pub struct CategoricalNode {
    potentials: Vec<f64>,
}

impl CategoricalNode {
    pub fn new(potentials: Vec<f64>) -> Self {
        CategoricalNode { potentials }
    }
}

impl NodePotential for CategoricalNode {
    fn n_values(&self) -> usize {
        self.potentials.len()
    }

    fn potential(&self, value: usize) -> f64 {
        self.potentials[value]
    }

    fn update_potentials(&self, updates: &[f64]) -> Self {
        let mut new_potentials = self
            .potentials
            .iter()
            .zip(updates.iter())
            .map(|(pot, up)| pot + up)
            .collect::<Vec<f64>>();
        // renormalize so that the maximum value is 0, for numerical stability
        let max = new_potentials
            .iter()
            .fold(std::f64::NEG_INFINITY, |a, &b| a.max(b));
        for entry in new_potentials.iter_mut() {
            *entry -= max;
        }
        CategoricalNode::new(new_potentials)
    }
}

pub struct CategoricalEdge {
    dim_1: usize,
    dim_2: usize,
    potentials: Vec<f64>, // potentials in row-major order
}

impl CategoricalEdge {
    pub fn from_grid(potentials_grid: &[Vec<f64>]) -> Option<Self> {
        let dim_1 = potentials_grid.len();
        if dim_1 == 0 {
            return None;
        }
        let dim_2 = potentials_grid[0].len();
        if dim_2 == 0 {
            return None;
        }
        let mut potentials = Vec::<f64>::new();
        for row in potentials_grid {
            if row.len() != dim_2 {
                return None;
            }
            potentials.extend(row.iter());
        }
        Some(CategoricalEdge {
            dim_1,
            dim_2,
            potentials,
        })
    }
}

impl EdgePotential for CategoricalEdge {
    fn n_values_1(&self) -> usize {
        self.dim_1
    }

    fn n_values_2(&self) -> usize {
        self.dim_2
    }

    fn potential(&self, v1: usize, v2: usize) -> f64 {
        assert!(v1 < self.dim_1);
        assert!(v2 < self.dim_2);
        self.potentials[v1 * self.dim_2 + v2]
    }

    fn transpose(&self) -> Self {
        let mut potentials = Vec::with_capacity(self.potentials.len());
        for v2 in 0..self.n_values_2() {
            for v1 in 0..self.n_values_1() {
                potentials.push(self.potential(v1, v2));
            }
        }
        CategoricalEdge {
            dim_1: self.dim_2,
            dim_2: self.dim_1,
            potentials,
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
    fn test_logsumexp() {
        assert_eq!(log_sum_exp(&vec![0.0, 0.0]), (2.0_f64).ln());
        assert!((log_sum_exp(&vec![1000.0, 0.0]) - 1000.0).abs() < 1e-6);
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
        assert_eq!(wedge.edge_potential(1, 0).unwrap().potential(0, 0), 0.5);
        assert!(wedge.edge_potential(0, 2).is_none());
        assert!(wedge.edge_potential(2, 0).is_none());
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

    #[test]
    fn test_marginalize_ising_1_edge() {
        use std::f64::consts::E;

        let node1 = IsingNode::new(1.0);
        let edge = IsingEdge::new(1.0);
        let node2 = IsingNode::new(1.0);
        let mut edge_set = BTreeMap::new();
        edge_set.insert((0, 1), edge);
        let graph = DiscreteUndirectedGraph::new(vec![node1, node2], edge_set);
        let marginals = graph.compute_marginals().unwrap();
        let expected_prob_ratio = (E.powi(4) + 1.0) / 2.0;
        let prob_ratio_0 = (marginals[0].potential(1) - marginals[0].potential(0)).exp();
        assert!((expected_prob_ratio - prob_ratio_0).abs() < 1e-7);
        let prob_ratio_1 = (marginals[1].potential(1) - marginals[1].potential(0)).exp();
        assert!((expected_prob_ratio - prob_ratio_1).abs() < 1e-7);
    }

    #[test]
    fn test_marginalize_ising_isolated() {
        let nodes = vec![IsingNode::new(-1.0), IsingNode::new(1.0)];
        let graph = DiscreteUndirectedGraph::<IsingNode, IsingEdge>::new(nodes, BTreeMap::new());
        let marginals = graph.compute_marginals().unwrap();
        assert_eq!(marginals[0].potential(1), -1.0);
        assert_eq!(marginals[1].potential(1), 1.0);
    }

    fn categorical_edge_2x3() -> CategoricalEdge {
        CategoricalEdge::from_grid(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap()
    }

    #[test]
    fn test_categorical_edge_potentials() {
        let edge = categorical_edge_2x3();
        assert_eq!(edge.n_values_1(), 2);
        assert_eq!(edge.n_values_2(), 3);
        assert_eq!(edge.potential(0, 0), 1.0);
        assert_eq!(edge.potential(0, 2), 3.0);
        assert_eq!(edge.potential(1, 0), 4.0);
        assert_eq!(edge.potential(1, 2), 6.0);
    }

    #[test]
    fn test_categorical_edge_transpose() {
        let edge = categorical_edge_2x3().transpose();
        assert_eq!(edge.n_values_1(), 3);
        assert_eq!(edge.n_values_2(), 2);
        assert_eq!(edge.potential(0, 0), 1.0);
        assert_eq!(edge.potential(2, 0), 3.0);
        assert_eq!(edge.potential(0, 1), 4.0);
        assert_eq!(edge.potential(2, 1), 6.0);
    }
}
