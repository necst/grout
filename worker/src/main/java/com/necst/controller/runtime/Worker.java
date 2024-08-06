package com.necst.controller.runtime;

import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.Enumeration;

public class Worker {
    public static void main(String[] args) {
        try {
            int port = 1090;
            if (args.length > 0)
                port = Integer.parseInt(args[0]);

            String ipAddress = getNonLoopbackIPAddress();
            System.setProperty("java.rmi.server.hostname", ipAddress);
            System.out.println("IP ADDR: "+ipAddress);

            Registry registry = LocateRegistry.createRegistry(port);
            registry.rebind("worker", new Servant());
            System.out.println("Worker is ready on port " + port + ".");
        } catch (Exception e) {
            System.out.println("Worker service error: " + e.getMessage());
        }
    }

    private static String getNonLoopbackIPAddress() throws SocketException {
        Enumeration<NetworkInterface> interfaces = NetworkInterface.getNetworkInterfaces();
        while (interfaces.hasMoreElements()) {
            NetworkInterface iface = interfaces.nextElement();
            Enumeration<InetAddress> addresses = iface.getInetAddresses();
            while (addresses.hasMoreElements()) {
                InetAddress addr = addresses.nextElement();
                if (!addr.isLoopbackAddress() && addr instanceof Inet4Address) {
                    return addr.getHostAddress();
                }
            }
        }
        throw new SocketException("Unable to determine non-loopback IP address");
    }
}