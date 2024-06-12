/*  
    N-Body simulation code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <errno.h>
#include "mpi.h"

#define GRAVITY     1.1
#define FRICTION    0.01
#define MAXBODIES   10000
#define DELTA_T     (0.025/5000)
#define BOUNCE      -0.9
#define SEED        27102015


struct bodyType {
    double x[2];        /* Old and new X-axis coordinates */
    double y[2];        /* Old and new Y-axis coordinates */
    double xf;          /* force along X-axis */
    double yf;          /* force along Y-axis */
    double xv;          /* velocity along X-axis */
    double yv;          /* velocity along Y-axis */
    double mass;        /* Mass of the body */
    double radius;      /* width (derived from mass) */
};


struct world {
    struct bodyType bodies[MAXBODIES];
    int                 bodyCt;
    int                 old;    // Flips between 0 and 1

    /*  Dimensions of space (very finite, ain't it?) */
    int                 xdim;
    int                 ydim;
};

/*  Macros to hide memory layout
*/
#define X(w, B)        (w)->bodies[B].x[(w)->old]
#define XN(w, B)       (w)->bodies[B].x[(w)->old^1]
#define Y(w, B)        (w)->bodies[B].y[(w)->old]
#define YN(w, B)       (w)->bodies[B].y[(w)->old^1]
#define XF(w, B)       (w)->bodies[B].xf
#define YF(w, B)       (w)->bodies[B].yf
#define XV(w, B)       (w)->bodies[B].xv
#define YV(w, B)       (w)->bodies[B].yv
#define R(w, B)        (w)->bodies[B].radius
#define M(w, B)        (w)->bodies[B].mass


static void
clear_forces(struct world *world, int start, int end)
{
    int b;

    /* Clear force accumulation variables */
    for (b = 0; b < world->bodyCt; ++b) {
        YF(world, b) = XF(world, b) = 0;
    }
}

static void
compute_forces(struct world *world, int start, int end, int bodies_per_node, int rank, int nodes)
{
    int b, c;
   double accumulateForces_x[world->bodyCt], accumulateForces_y[world->bodyCt];
    memset(accumulateForces_y, 0, sizeof(accumulateForces_y));
    memset(accumulateForces_x, 0, sizeof(accumulateForces_x));
    
    /* Incrementally accumulate forces from each body pair,
       skipping force of body on itself (c == b)
    */
    for (b = start; b < end; ++b) {

 
        for (c = b + 1; c < world->bodyCt; ++c) {
              
            double dx = X(world, c) - X(world, b);
            double dy = Y(world, c) - Y(world, b);
            double angle = atan2(dy, dx);
            double dsqr = dx*dx + dy*dy;
            double mindist = R(world, b) + R(world, c);
            double mindsqr = mindist*mindist;
            double forced = ((dsqr < mindsqr) ? mindsqr : dsqr);
            double force = M(world, b) * M(world, c) * GRAVITY / forced;
            double xf = force * cos(angle);
            double yf = force * sin(angle);

            /* Slightly sneaky...
               force of b on c is negative of c on b;
            */
            XF(world, b) += xf;
            YF(world, b) += yf;
            XF(world, c) -= xf;
            YF(world, c) -= yf;
        
        // if body c is outside current process, accumulate it's x 
        // and y forces so that it can be added later. 
            if(c >= end){
            accumulateForces_x[c] -= xf;
            accumulateForces_y[c] -= yf;
            
            }

        }
    }
// send the accumulated forces
MPI_Allreduce(MPI_IN_PLACE, accumulateForces_x,world->bodyCt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
MPI_Allreduce(MPI_IN_PLACE, accumulateForces_y,world->bodyCt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

// add the accumulated forces to the final result.
for (int i = start ; i< end ; ++i){
    XF(world , i) += accumulateForces_x[i];
   YF(world , i) += accumulateForces_y[i];
//    printf("body: %d , aFx: %lf    ,   aFy: %lf\n", i,accumulateForces_x[i], accumulateForces_y[i]);
   
//    memset(accumulateForces_y, 0, sizeof(accumulateForces_y));
    
}
}

static void
compute_velocities(struct world *world, int start, int end, double* buf)
{
    int b;
    int j = 0;
    for (b = start; b < end; ++b) {
        double xv = XV(world, b);
        double yv = YV(world, b);
        double force = sqrt(xv*xv + yv*yv) * FRICTION;
        double angle = atan2(yv, xv);
        double xf = XF(world, b) - (force * cos(angle));
        double yf = YF(world, b) - (force * sin(angle));

        XV(world, b) += (xf / M(world, b)) * DELTA_T;
        YV(world, b) += (yf / M(world, b)) * DELTA_T;

        //store data in buf for sending via allgather
        buf[2+j] = XV(world, b);
        buf[3+j] = YV(world, b);
    }
}

static void
compute_positions(struct world *world, int start, int end, double* buf)
{
    int b;
    int j = 0;
    for (b = start; b < end; ++b) {
        double xn = X(world, b) + XV(world, b) * DELTA_T;
        double yn = Y(world, b) + YV(world, b) * DELTA_T;

        /* Bounce off image "walls" */
        if (xn < 0) {
            xn = 0;
            XV(world, b) = -XV(world, b);
        } else if (xn >= world->xdim) {
            xn = world->xdim - 1;
            XV(world, b) = -XV(world, b);
        }
        if (yn < 0) {
            yn = 0;
            YV(world, b) = -YV(world, b);
        } else if (yn >= world->ydim) {
            yn = world->ydim - 1;
            YV(world, b) = -YV(world, b);
        }

        /* Update position */
        XN(world, b) = xn;
        YN(world, b) = yn;

        // store in buf to send via allgather
        buf[0+j] = xn;
        buf[1+j] = yn;
        j=j+4;
    }
}


/*  Graphic output stuff...
 */

#include <fcntl.h>
#include <sys/mman.h>

struct filemap {
    int            fd;
    off_t          fsize;
    void          *map;
    unsigned char *image;
};


static void
filemap_close(struct filemap *filemap)
{
    if (filemap->fd == -1) {
        return;
    }
    close(filemap->fd);
    if (filemap->map == MAP_FAILED) {
        return;
    }
    munmap(filemap->map, filemap->fsize);
}


static unsigned char *
Eat_Space(unsigned char *p)
{
    while ((*p == ' ') ||
           (*p == '\t') ||
           (*p == '\n') ||
           (*p == '\r') ||
           (*p == '#')) {
        if (*p == '#') {
            while (*(++p) != '\n') {
                // skip until EOL
            }
        }
        ++p;
    }

    return p;
}


static unsigned char *
Get_Number(unsigned char *p, int *n)
{
    p = Eat_Space(p);  /* Eat white space and comments */

    int charval = *p;
    if ((charval < '0') || (charval > '9')) {
        errno = EPROTO;
        return NULL;
    }

    *n = (charval - '0');
    charval = *(++p);
    while ((charval >= '0') && (charval <= '9')) {
        *n *= 10;
        *n += (charval - '0');
        charval = *(++p);
    }

    return p;
}


static int
map_P6(const char *filename, int *xdim, int *ydim, struct filemap *filemap)
{
    /* The following is a fast and sloppy way to
       map a color raw PPM (P6) image file
    */
    int maxval;
    unsigned char *p;

    /* First, open the file... */
    if ((filemap->fd = open(filename, O_RDWR)) < 0) {
        goto ppm_abort;
    }

    /* Read size and map the whole file... */
    filemap->fsize = lseek(filemap->fd, (off_t)0, SEEK_END);
    filemap->map = mmap(0,                      // Put it anywhere
                        filemap->fsize,         // Map the whole file
                        (PROT_READ|PROT_WRITE), // Read/write
                        MAP_SHARED,             // Not just for me
                        filemap->fd,            // The file
                        0);                     // Right from the start
    if (filemap->map == MAP_FAILED) {
        goto ppm_abort;
    }

    /* File should now be mapped; read magic value */
    p = filemap->map;
    if (*(p++) != 'P') goto ppm_abort;
    switch (*(p++)) {
    case '6':
        break;
    default:
        errno = EPROTO;
        goto ppm_abort;
    }

    p = Get_Number(p, xdim);            // Get image width */
    if (p == NULL) goto ppm_abort;
    p = Get_Number(p, ydim);            // Get image width */
    if (p == NULL) goto ppm_abort;
    p = Get_Number(p, &maxval);         // Get image max value */
    if (p == NULL) goto ppm_abort;

    /* Should be 8-bit binary after one whitespace char... */
    if (maxval > 255) {
        goto ppm_abort;
    }
    if ((*p != ' ') &&
        (*p != '\t') &&
        (*p != '\n') &&
        (*p != '\r')) {
        errno = EPROTO;
        goto ppm_abort;
    }

    /* Here we are... next byte begins the 24-bit data */
    filemap->image = p + 1;

    return 0;

ppm_abort:
    filemap_close(filemap);

    return -1;
}


static inline void
color(const struct world *world, unsigned char *image, int x, int y, int b)
{
    unsigned char *p = image + (3 * (x + (y * world->xdim)));
    int tint = ((0xfff * (b + 1)) / (world->bodyCt + 2));

    p[0] = (tint & 0xf) << 4;
    p[1] = (tint & 0xf0);
    p[2] = (tint & 0xf00) >> 4;
}

static inline void
black(const struct world *world, unsigned char *image, int x, int y)
{
    unsigned char *p = image + (3 * (x + (y * world->xdim)));

    p[2] = (p[1] = (p[0] = 0));
}

static void
display(const struct world *world, unsigned char *image)
{
    double i, j;
    int b;

    /* For each pixel */
    for (j = 0; j < world->ydim; ++j) {
        for (i = 0; i < world->xdim; ++i) {
            /* Find the first body covering here */
            for (b = 0; b < world->bodyCt; ++b) {
                double dy = Y(world, b) - j;
                double dx = X(world, b) - i;
                double d = sqrt(dx*dx + dy*dy);

                if (d <= R(world, b)+0.5) {
                    /* This is it */
                    color(world, image, i, j, b);
                    goto colored;
                }
            }

            /* No object -- empty space */
            black(world, image, i, j);

colored:        ;
        }
    }
}

static void
print(struct world *world)
{
    int b;

    for (b = 0; b < world->bodyCt; ++b) {
        printf("%10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
               X(world, b), Y(world, b), XF(world, b), YF(world, b), XV(world, b), YV(world, b));
    }
}

static long long
nr_flops(int n, int steps) {
  long long nr_flops = 0;
  // compute forces
  nr_flops += 20 * (n * (n-1) / 2);
  // compute velocities
  nr_flops += 18 * n;
  // compute positions
  nr_flops += 4 * n;

  nr_flops *= steps;

  return nr_flops;
}
    

/*  Main program...
*/

int rank;
int nodes;
// MPI_Datatype aggregateType;

int
main(int argc, char **argv)
{
    unsigned int lastup = 0;
    unsigned int secsup;
    int b;
    int steps;
    double rtime;
    struct timeval start;
    struct timeval end;
    struct filemap image_map;

    struct world *world = calloc(1, sizeof *world);
    if (world == NULL) {
        fprintf(stderr, "Cannot calloc(world)\n");
        exit(1);
    }

    /* Get Parameters */
    // if (argc != 5) {
    //     fprintf(stderr, "Usage: %s num_bodies secs_per_update ppm_output_file steps\n",
    //             argv[0]);
    //     exit(1);
    // }
    if ((world->bodyCt = atol(argv[1])) > MAXBODIES ) {
        fprintf(stderr, "Using only %d bodies...\n", MAXBODIES);
        world->bodyCt = MAXBODIES;
    } else if (world->bodyCt < 2) {
        fprintf(stderr, "Using two bodies...\n");
        world->bodyCt = 2;
    }
    secsup = atoi(argv[2]);
    if (map_P6(argv[3], &world->xdim, &world->ydim, &image_map) == -1) {
        fprintf(stderr, "Cannot read %s: %s\n", argv[3], strerror(errno));
        exit(1);
    }
    steps = atoi(argv[4]);
  
    /*  Initialize MPI  and shared variables*/
    
    MPI_Init(& argc, & argv);
    MPI_Comm_size(MPI_COMM_WORLD, & nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, & rank);
 
  if(rank == 0 ){
    fprintf(stderr, "Running N-body with %i bodies and %i steps\n", world->bodyCt, steps);
    }
    
    int bodies_per_node = (world->bodyCt) / nodes;
    int startIndex = rank * bodies_per_node;
    int endIndex = startIndex + bodies_per_node ;
   
    /* Initialize simulation data in proc0 */
    // need to bcast this to all nodes

    //array to hold all the bcast data
    int bcastDataSize =  world->bodyCt * 5;
    double bcastData[bcastDataSize];    // x[0], y[0], xv, yv, mass

    // array to hold allgather data - x[0], y[0], xv, yv
    int rbuf_size = 4 * world->bodyCt;      // rbuf - receive buffer
    int sbuf_size = 4 * bodies_per_node;    // sbuf - send buffer
    double rbuf[rbuf_size];    // contains x[0], y[0], xv, yv of all bodies, all nodes, in that step
    double sbuf[sbuf_size];   // contains x[0], y[0], xv, yv of all bodies of that node, in that step
     

    if (rank == 0){ 
    srand(SEED);
    int j=0;
    for (b = 0; b < world->bodyCt; ++b) {
        X(world, b) = (rand() % world->xdim);
        Y(world, b) = (rand() % world->ydim);
        R(world, b) = 1 + ((b*b + 1.0) * sqrt(1.0 * ((world->xdim * world->xdim) + (world->ydim * world->ydim)))) /
                (25.0 * (world->bodyCt*world->bodyCt + 1.0));
        M(world, b) = R(world, b) * R(world, b) * R(world, b);

        XV(world, b) = ((rand() % 20000) - 10000) / 2000.0;
        YV(world, b) = ((rand() % 20000) - 10000) / 2000.0;
        
        bcastData[j+0] =  X(world, b);
        bcastData[j+1] =  Y(world, b);
        bcastData[j+2] =  XV(world, b);
        bcastData[j+3] =  YV(world, b);
        bcastData[j+4] =  M(world, b);    
        j=j+5;

    }
    } 
   // Bcast initial data
    MPI_Bcast(bcastData, bcastDataSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
// distribute data from the bcast array to the proper variables
    if (rank != 0){
        
        int j=0;
        int i;
         for (i = 0; i < world->bodyCt; ++i){
            X(world, i)     = bcastData[j+0]  ;
            Y(world, i)     = bcastData[j+1]   ;
            XV(world, i)    = bcastData[j+2]   ;
            YV(world, i)    = bcastData[j+3]   ;
            M(world, i)     = bcastData[j+4]   ; 
            j=j+5;           
         }                 
    }
 
  // calulate radius on each node instead of bcasting it...
  // maybe its faster than sending?
   if (rank != 0){
    for (b = 0; b < world->bodyCt; ++b){
         R(world, b) = 1 + ((b*b + 1.0) * sqrt(1.0 * ((world->xdim * world->xdim) + (world->ydim * world->ydim)))) /
                (25.0 * (world->bodyCt*world->bodyCt + 1.0));
        }
   }
   
    //start timing
    double startTime = MPI_Wtime();

    /* Main Loop */
    for (int step = 0; step < steps; step++) {
        clear_forces(world, startIndex, endIndex);       
        compute_forces(world, startIndex, endIndex, bodies_per_node, rank, nodes);              
        compute_velocities(world, startIndex, endIndex, sbuf);
        compute_positions(world, startIndex, endIndex, sbuf);
    
        // gather newly computed data
        MPI_Allgather(sbuf, sbuf_size, MPI_DOUBLE, rbuf, sbuf_size, MPI_DOUBLE, MPI_COMM_WORLD);
      
        // assign the received data from rbuf to the proper variables
        int j =0;
        for(int i = 0; i < world->bodyCt; ++i){
            XN(world, i)     =    rbuf[0+j];
            YN(world, i)     =    rbuf[1+j];
            j=j+4;
        }
         // printf("p%d val: %lf\n", rank, XF(world, 8));
     /* Flip old & new coordinates */
          world->old ^= 1;
    }
   
    double endTime = MPI_Wtime();
// only process 0 will print final values, so i communicate final force
// and velocity values from all proceses to process 0 now 

// finalForces is the send buffer
    int finalForcesSize = 4 * bodies_per_node;
   double finalForces[finalForcesSize];
   
// finalForcesRecv is the receive buffer
    int finalForcesRecvSize = 4 * world->bodyCt;
    double finalForcesRecv[finalForcesRecvSize] ;

    
    for(int i = 0; i < nodes ; ++i){
      int j = 0;
        if(rank == i){
            for(int k = startIndex; k < endIndex ; ++k){
                finalForces[0+j] = XF(world, k);
                finalForces[1+j] = YF(world, k);
                finalForces[2+j] = XV(world, k);
                finalForces[3+j] = YV(world, k);
               
                j=j+4;
            }
        }
    }

    MPI_Gather(finalForces, finalForcesSize, MPI_DOUBLE, finalForcesRecv, finalForcesSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   // now set gathered data from recv buffer to proper variables in process 0
    if (rank == 0){
        int j = 0;
     
        for (int i = 0 ; i < world->bodyCt ; ++i){
        XF(world, i)    =   finalForcesRecv[0+j];
        YF(world, i)    =   finalForcesRecv[1+j];
        XV(world, i)    =   finalForcesRecv[2+j];
        YV(world, i)    =   finalForcesRecv[3+j];
        j = j + 4;
        }
    print(world);
    fprintf(stderr, "\nN-body took: %.3f seconds\n", endTime - startTime);
    fprintf(stderr, "Performance N-body: %.2f GFLOPS\n", nr_flops(world->bodyCt, steps) / 1e9 / rtime);
    }   
    free(world);
    MPI_Finalize();
    return 0;
}
