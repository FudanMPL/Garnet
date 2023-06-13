#include "Processor/Binary_File_IO.h"

/* 
 * Provides generalised file read and write methods for arrays of shares.
 * Stateless and not optimised for multiple reads from file.
 * Intended for application specific file IO.
 */

inline string Binary_File_IO::filename(int my_number)
{
  string dir = "Persistence";
  mkdir_p(dir.c_str());
  return dir + "/Transactions-P" + to_string(my_number) + ".data";
}

template<class T> 

void Binary_File_IO::write_to_file(const string filename,
    const vector<T>& buffer, long start_pos)
{
  ofstream outf;

  outf.open(filename, ios::out | ios::binary | ios::ate | ios::in);
  if (outf.fail()) { throw file_error(filename); }

  if (start_pos != -1)
    {
      long write_pos = file_signature<T>().get_total_length() + start_pos * T::size();
      // fill with zeros if needed
      for (long i = outf.tellp(); i < write_pos; i++)
        outf.put(0);
      outf.seekp(write_pos);
    }

  for (unsigned int i = 0; i < buffer.size(); i++)
  {
    buffer[i].output(outf, false);
  }

  if (outf.fail())
    throw runtime_error("failed writing to " + filename);

  outf.close();
}

template<class T>
void Binary_File_IO::read_from_file(const string filename, vector< T >& buffer, const int start_posn, int &end_posn)
{
  ifstream inf;
  inf.open(filename, ios::in | ios::binary);
  if (inf.fail()) { throw file_missing(filename, "Binary_File_IO.read_from_file expects this file to exist."); }

  check_file_signature<T>(inf, filename).get_length();
  auto data_start = inf.tellg();

  int size_in_bytes = T::size() * buffer.size();
  int n_read = 0;
  char read_buffer[size_in_bytes];
  inf.seekg(start_posn * T::size(), iostream::cur);
  do
  {
      inf.read(read_buffer + n_read, size_in_bytes - n_read);
      n_read += inf.gcount();

      if (inf.eof())
      {
        stringstream ss;
        ss << "Got to EOF when reading from disk (expecting " << size_in_bytes
            << " bytes from " << (long(data_start) + start_posn * T::size())
            << ").";
        throw file_error(ss.str());
      }
      if (inf.fail())
      {
        stringstream ss;
        ss << "IO problem when reading from disk";
        throw file_error(ss.str());
      }
  }
  while (n_read < size_in_bytes);

  end_posn = (inf.tellg() - data_start) / T::size();
  assert (end_posn == start_posn + int(buffer.size()));

  //Check if at end of file by getting 1 more char.
  inf.get();
  if (inf.eof())
    end_posn = -1;
  inf.close();

  for (unsigned int i = 0; i < buffer.size(); i++)
    buffer[i].assign(&read_buffer[i*T::size()]);
}
