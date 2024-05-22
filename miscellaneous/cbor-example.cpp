#include <cbor.h>
#include <stdio.h>
#include <string>

int main(void) {
  cbor_item_t* root = cbor_new_indefinite_array();
	bool success = true;

	cbor_item_t* b = cbor_new_definite_map(400);
	for (int i = 0; i < 400; i++) {
		const double value = rand() / (double)RAND_MAX;
		const std::string key = std::to_string(i);
		success &= cbor_map_add(
				b, (struct cbor_pair){
									.key = cbor_move(cbor_build_string(key.c_str())),
									.value = cbor_move(cbor_build_float8(value))});
	}
	success &= cbor_array_push(root, b);

	for (int i = 0; i < 4000; i++) {
		cbor_item_t* c = cbor_new_definite_map(4);
		success &= cbor_map_add(
				c, (struct cbor_pair){
									.key = cbor_move(cbor_build_string("a")),
									.value = cbor_move(cbor_build_float8(rand() / (double)RAND_MAX))});
		success &= cbor_map_add(
				c, (struct cbor_pair){
									.key = cbor_move(cbor_build_string("b")),
									.value = cbor_move(cbor_build_float8(2.71828182845904523536))});

		cbor_item_t* d = cbor_new_definite_array(3);
		for (int i = 0; i < 3; i++) {
			const double value = rand() / (double)RAND_MAX;
			success &= cbor_array_push(d, cbor_move(cbor_build_float8(value)));
		}
		success &= cbor_map_add(
				c, (struct cbor_pair){
									.key = cbor_move(cbor_build_string("position")),
									.value = d});

		cbor_item_t* e = cbor_new_definite_array(4);
		for (int i = 0; i < 4; i++) {
			const double value = rand() / (double)RAND_MAX;
			success &= cbor_array_push(e, cbor_move(cbor_build_float8(value)));
		}
		success &= cbor_map_add(
				c, (struct cbor_pair){
									.key = cbor_move(cbor_build_string("position")),
									.value = e});							

		success &= cbor_array_push(root, c);
	}

	cbor_item_t* d = cbor_new_definite_array(10000);
	for (int i = 0; i < 10000; i++) {
		const double value = rand() / (double)RAND_MAX;
		success &= cbor_array_push(d, cbor_move(cbor_build_float8(value)));
	}
	success &= cbor_array_push(root, d);

	cbor_item_t* e = cbor_new_definite_array(3723);
	for (int i = 0; i < 3723; i++) {
		const double value = rand() / (double)RAND_MAX;
		success &= cbor_array_push(e, cbor_move(cbor_build_float8(value)));
	}
	success &= cbor_array_push(root, e);

  if (!success) return 1;

  /* Output: `length` bytes of data in the `buffer` */
  unsigned char* buffer;
  size_t buffer_size;
  cbor_serialize_alloc(root, &buffer, &buffer_size);

	// Write the buffer to the file "out.bin"
	FILE *f = fopen("out.bin", "wb");
	fwrite(buffer, 1, buffer_size, f);
	fclose(f);
	free(buffer);

  // fwrite(buffer, 1, buffer_size, stdout);
  // free(buffer);

  // fflush(stdout);
  // cbor_decref(&root);
}
