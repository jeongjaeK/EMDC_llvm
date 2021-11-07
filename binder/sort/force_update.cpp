#include <iostream>
#include <elfio/elfio.hpp>

using namespace ELFIO;

int main( int argc, char** argv ){
	if ( argc != 2 ){
		std::cout << "Usage : update-section <elf_file>" << std::endl;
		return 1;
	}

	elfio reader;

	if ( !reader.load( argv[1] ) ){
		std::cout << "Failed to load ELF file" << argv[1] << std::endl;
		return 2;
	}

	section* psection = reader.sections[1]; //__CLANG_OFFLOAD_BUNDLE__sycl-spir64_fpga
	auto sec_size = psection->get_size();
//	auto sec_flag = psection->get_flags() ^ SHF_ALLOC;
//	psection->set_flags(sec_flag);
	psection->set_size(0);

	elf_header* pheader = reader.header;
	auto sh_offset = pheader->get_sections_offset() - sec_size;
	pheader->set_sections_offset(sh_offset);
//	pheader->set_sections_offset(0x239e2);

	if ( reader.save("test") ) {
		std::cout << "Successfully saved." << std::endl;	
	}
	else{
		std::cout << "Failed to save." << std::endl;	
	}
	
	
	
	
	return 0;	
}
