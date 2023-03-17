library("maps")

# create .RData file from fluorescence .nc files
allfnames <- list.files(
    path="data/may_16_31_2017",
    pattern="\\.nc",
    full.names=TRUE,
    include.dirs=TRUE
)
usa <- map("usa")
firstna <- which(is.na(usa$x))[1]

usax <- usa$x[1:(firstna-1)]
usay <- usa$y[1:(firstna-1)]
plot(usax, usay)

flr <- c()
lon <- c()
lat <- c()
tim <- c()

for(j in 1:length(allfnames)) {
    
    fname <- allfnames[j]
    
    nc <- ncdf4::nc_open(fname)
    # flrj <- ncdf4::ncvar_get(nc, "SIF_757nm")
    # lonj <- ncdf4::ncvar_get(nc, "longitude")
    # latj <- ncdf4::ncvar_get(nc, "latitude")
    # timj <- ncdf4::ncvar_get(nc, "time")
    flrj <- ncdf4::ncvar_get(nc, "Daily_SIF_757nm")
    lonj <- ncdf4::ncvar_get(nc, "Longitude")
    latj <- ncdf4::ncvar_get(nc, "Latitude")
    timj <- ncdf4::ncvar_get(nc, "Delta_Time")
    
    in_usa <- sp::point.in.polygon(lonj, latj, usax, usay) == 1
    
    flr <- c(flr, flrj[in_usa])
    lon <- c(lon, lonj[in_usa])
    lat <- c(lat, latj[in_usa])
    tim <- c(tim, timj[in_usa])
}

dups <- duplicated( cbind(lon,lat) )
flr <- flr[!dups]
lon <- lon[!dups]
lat <- lat[!dups]
tim <- tim[!dups]

# save(flr, lon, lat, tim, file="data/aug_01_31_2018_fluorescence.RData")
data = data.frame(
    lon=lon,
    lat=lat,
    time=tim,
    flr=flr
)
write.csv(
    data,
    quote=FALSE,
    row.names=FALSE,
    # file="data/aug_01_31_2018_fluorescence.csv"
    # file="data/may_16_31_2017_fluorescence.csv"
    file="fluorescence_data.csv"
)

