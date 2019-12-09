#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
double  wtime(void);

int calc_nrows(int rank, int size, int nx);

int main(int argc, char* argv[])
{

  int rank;              /* the rank of this process */
  int up;              /* the rank of the process to the left */
  int down;             /* the rank of the process to the right */
  int size;              /* number of processes in the communicator */
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */
  int local_nrows;       /* number of rows apportioned to this rank */
  int local_ncols;       /* number of columns apportioned to this rank */

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  /*
  ** MPI_Init returns once it has started up processes
  ** Get size of cohort and rank for this process
  */
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  /*
  ** determine process ranks to the left and right of rank
  ** respecting periodic boundary conditions
  */
  up = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  down = (rank + 1) % size;

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // we pad the outer edge of the image to avoid out of range address issues in
  // stencil
  int width = nx + 2;
  int height = ny + 2;

  /*
  ** determine local grid size
  ** each rank gets all the rows, but a subset of the number of columns
  */
  local_nrows = calc_nrows(rank, size, nx);
  local_ncols = ny;
  if (local_nrows < 1) {
    fprintf(stderr,"Error: too many processes:- local_ncols < 1\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Allocate the image
  float* image = malloc(sizeof(float) * width * height);
  float* tmp_image = malloc(sizeof(float) * width * height);

  // Set the input image
  init_image(nx, ny, width, height, image, tmp_image);

  printf("init image done\n");

  // Allocate local image
  float* local = malloc(sizeof(float) * (local_ncols + 2) * (local_nrows + 2));
  float* tmp_local = malloc(sizeof(float) * (local_ncols + 2) * (local_nrows + 2));

  //Initialise local tmp_image
  int step = nx/size;
  for (int i = 0; i < local_nrows + 2; i++) {
    for (int j = 0; j < local_ncols + 2; j++) {
      if (j > 0 && j < (local_ncols + 1) && i > 0 && i < local_nrows + 1) {
	       local[j + i * (local_ncols + 2)] = image[j + (i * width) + (step * rank * width)];
         tmp_local[j + i * (local_ncols + 2)] = image[j + (i * width) + (step * rank * width)];
       }               /* core cells */
      else {
	       local[j + i * (local_ncols + 2)] = 0.0f;
         tmp_local[j + i * (local_ncols + 2)] = 0.0f;
       }                      /* halo cells */
    }
  }

  printf("init local image done for rank = %d\n", rank);

  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {

    //Halo exchange for local
    //Send up, receive down
    MPI_Sendrecv(&local[local_ncols + 2 + 1], local_ncols, MPI_FLOAT, up, tag,
      &local[(local_nrows + 1) * (local_ncols + 2) + 1], local_ncols, MPI_FLOAT, down, tag, MPI_COMM_WORLD, &status);
    //Send down, receive up
    MPI_Sendrecv(&local[(local_nrows) * (local_ncols + 2) + 1], local_ncols, MPI_FLOAT, down, tag,
      &local[1], local_ncols, MPI_FLOAT, up, tag, MPI_COMM_WORLD, &status);

    printf("send recv for local done in rank = %d\n", rank);

    stencil(local_nrows, local_ncols, width, height, local, tmp_local);

    //Halo exchange for temp
    //Send up, receive down
    MPI_Sendrecv(&tmp_local[local_ncols + 2 + 1], local_ncols, MPI_FLOAT, up, tag,
      &tmp_local[(local_nrows + 1) * (local_ncols + 2) + 1], local_ncols, MPI_FLOAT, down, tag, MPI_COMM_WORLD, &status);
    //Send down, receive up
    MPI_Sendrecv(&tmp_local[(local_nrows) * (local_ncols + 2) + 1], local_ncols, MPI_FLOAT, down, tag,
      &tmp_local[1], local_ncols, MPI_FLOAT, up, tag, MPI_COMM_WORLD, &status);
    stencil(local_nrows, local_ncols, width, height, tmp_local, local);
  }
  double toc = wtime();

  if (rank == 0) {
    // Output
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc - tic);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, nx, ny, width, height, image);
  }

  MPI_Finalize();

  free(image);
  free(tmp_image);
  free(local);
  free(tmp_local);

}

void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image)
{
  for (int i = 1; i < ny + 1; ++i) {
    for (int j = 1; j < nx + 1; ++j) {
	     int temp = j + i * (ny + 2);
       tmp_image[temp] = image[temp] * 0.6f + 0.1f * (image[temp - (ny + 2)] + image[temp + (ny + 2)] + image[temp - 1] + image[temp + 1]);
    }
  }
}

// Create the input image
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image)
{
  // Zero everything
  for (int j = 0; j < ny + 2; ++j) {
    for (int i = 0; i < nx + 2; ++i) {
      image[j + i * height] = 0.0;
      tmp_image[j + i * height] = 0.0;
    }
  }

  const int tile_size = 64;
  // checkerboard pattern
  for (int jb = 0; jb < ny; jb += tile_size) {
    for (int ib = 0; ib < nx; ib += tile_size) {
      if ((ib + jb) % (tile_size * 2)) {
        const int jlim = (jb + tile_size > ny) ? ny : jb + tile_size;
        const int ilim = (ib + tile_size > nx) ? nx : ib + tile_size;
        for (int j = jb + 1; j < jlim + 1; ++j) {
          for (int i = ib + 1; i < ilim + 1; ++i) {
            image[j + i * height] = 100.0;
          }
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image)
{
  // Open output file
  FILE* fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      if (image[j + i * height] > maximum) maximum = image[j + i * height];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      fputc((char)(255.0 * image[j + i * height] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// Get the current time in seconds since the Epoch
double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int calc_nrows(int rank, int size, int nx)
{
  int nrows;

  nrows = nx / size;       /* integer division */
  if ((nx % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      nrows += nx % size;  /* add remainder to last rank */
  }

  return nrows;

}
