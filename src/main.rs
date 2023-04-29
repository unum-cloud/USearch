#[cxx::bridge]
mod ffi {
    
    // Shared structs with fields visible to both languages.
    struct Results {
        size: usize,
        labels: Vec<u32>,
        distances: Vec<f32>,
    }

    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("rust.h");

        type Index;

        fn new_index() -> UniquePtr<Index>;

        fn dim(&self) -> u64;
        fn connectivity(&self) -> u64;
        fn size(&self) -> u64;
        fn capacity(&self) -> u64;

        fn add(&self, label: u64, vector: &[f32]);
        fn search(&self, query: &[f32], count: u64) -> Results;
    }
}


fn main() {
    let client = ffi::new_index();

    // Upload a blob.
    let chunks = vec![b"fearless".to_vec(), b"concurrency".to_vec()];
    let mut buf = MultiBuf { chunks, pos: 0 };
    let blobid = client.add(&mut buf);
    println!("blobid = {}", blobid);

    // Read back the tags.
    let search = client.search(blobid);
    println!("tags = {:?}", search.tags);
}