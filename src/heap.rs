#[derive(Default)]
pub struct HeapPQ<T: PartialOrd> {
    data: Vec<T>,
}

impl<T: PartialOrd> HeapPQ<T> {
    pub fn new() -> Self {
        HeapPQ { data: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn sift_up(&mut self, index: usize) {
        /* Propagate the element at `index` up through the heap as needed */
        if index > 0 {
            let parent_index = (index - 1) / 2;
            if self.data[index] > self.data[parent_index] {
                self.data.swap(index, parent_index);
                self.sift_up(parent_index)
            }
        }
    }

    fn sift_down(&mut self, index: usize) {
        /* Fix the heap property when a new element is at the front */
        let len = self.data.len();
        let child_1_idx = 2 * index + 1;
        let child_2_idx = 2 * index + 2;
        if child_2_idx < len {
            // swap with larger child
            if self.data[child_1_idx] > self.data[child_2_idx] {
                if self.data[child_1_idx] > self.data[index] {
                    self.data.swap(index, child_1_idx);
                    self.sift_down(child_1_idx)
                }
            } else if self.data[child_2_idx] > self.data[index] {
                self.data.swap(index, child_2_idx);
                self.sift_down(child_2_idx)
            }
        } else if child_1_idx < len && self.data[child_1_idx] > self.data[index] {
            self.data.swap(index, child_1_idx);
            self.sift_down(child_1_idx)
        }
    }

    pub fn push(&mut self, value: T) {
        self.data.push(value);
        self.sift_up(self.data.len() - 1);
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let len = self.data.len();
        self.data.swap(0, len - 1);
        let result = self.data.pop();
        self.sift_down(0);
        result
    }
}

impl<T: PartialOrd> From<Vec<T>> for HeapPQ<T> {
    fn from(v: Vec<T>) -> Self {
        let mut heap = HeapPQ { data: v };
        for i in (0..heap.data.len()).rev() {
            heap.sift_down(i);
        }
        heap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_pop_gets_largest() {
        let mut h = HeapPQ::from(vec![1, 2, 3, 4, 5]);
        assert_eq!(h.pop(), Some(5));
        assert_eq!(h.pop(), Some(4));
        assert_eq!(h.pop(), Some(3));
        assert_eq!(h.pop(), Some(2));
        assert_eq!(h.pop(), Some(1));
    }

    #[test]
    fn mixed_pushes_and_pops() {
        let mut h = HeapPQ::new();
        h.push(1);
        h.push(2);
        assert_eq!(h.pop(), Some(2));
        h.push(4);
        h.push(3);
        assert_eq!(h.pop(), Some(4));
        h.push(-1);
        h.push(-1);
        assert_eq!(h.pop(), Some(3));
    }
}
